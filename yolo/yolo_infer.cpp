#include "yolo_infer.hpp"
#include <trt/trt.hpp>
#include "preprocess.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <ctime>

template <typename ty>
static inline ty upbound_(ty val, ty n)
{
    return (val - 1 + n) / n * n;
}
static const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};

static float bbox_iou(float left1, float top1, float right1, float bottom1, float left2, float top2, float right2, float bottom2)
{
    float inter_left = std::max(left1, left2);
    float inter_top = std::max(top1, top2);
    float inter_right = std::min(right1, right2);
    float inter_bottom = std::min(bottom1, bottom2);

    float inter_area = std::max(inter_bottom - inter_top, 0.0f) * std::max(inter_right - inter_left, 0.0f);
    if (inter_area == 0.0f)
        return 0.0f;
    float union_area = (right1 - left1) * (bottom1 - top1) + (right2 - left2) * (bottom2 - top2) - inter_area;
    return inter_area / union_area; 
}

namespace tinycv
{
    namespace yolo
    {
        float bbox_iou(tinycv::yolo::Box &box1, tinycv::yolo::Box &box2)
        {
            return ::bbox_iou(box1.left, box1.top, box1.right, box1.bottom, box2.left, box2.top, box2.right, box2.bottom);
        }

        class YoloInferImpl: public YoloInfer
        {
        public:
            YoloInferImpl() = default;

            bool load(const std::string &engine_file, float confidence_threshold, float nms_threshold)
            {
                infer_ = trt::load_infer_from_file(engine_file);
                if (infer_ == nullptr)
                    return false;
                // infer_->print();
                confidence_threshold_ = confidence_threshold;
                nms_threshold_ = nms_threshold_;
                std::vector<int> input_dim = infer_->get_binding_dims(0);
                bbox_dims_ = infer_->get_binding_dims(1);
                // TODO: To support more version.
                n_classes_ = bbox_dims_[2] - 5;
                n_channels_ = input_dim[1];
                height_ = input_dim[2];
                width_ = input_dim[3];
                is_dynamic_mode_ = infer_->is_dynamic();
                return true;
            }

            int get_height() override { return height_; }
            int get_width() override { return width_; };
            int get_n_channels() override { return n_channels_; };
            int get_n_classes() override { return n_classes_; }

            BoxArray forward(Frame &frame, void *stream) override
            {
                std::vector<Frame> frames = {frame};
                std::vector<BoxArray> ret = forward(frames, stream);
                if (ret.empty())
                    return {};
                return ret.front();
            }

            int get_n_infer_batch(int n_input_batch)
            {
                // make sure the infer batch.
                std::vector<int> input_dims = infer_->get_binding_dims(0);  // the input dims.
                int n_infer_batch = input_dims[0];  // the number of batch trt supported.
                if (n_input_batch != n_infer_batch)  // the number of input batch != the number of batch trt supported. 
                {
                    if (is_dynamic_mode_)  // when support dynamic mode.
                    {
                        n_infer_batch = n_input_batch;
                        input_dims[0] = n_infer_batch;
                        if (!infer_->set_bindings_dims(0, input_dims))  // try to set the number of infer batch to the number of input batch
                            return 0;
                    }
                    else
                    {
                        if (n_infer_batch < n_input_batch)  // when static mode, requires the number of infer batch is greater equal than the number of input batch
                        {
                            INFO_ERROR(
                                "When using static model to infer, the number of input(%d) should not be greater than "
                                "the number of maximum batch(%d)", n_input_batch, n_infer_batch
                            );
                            return 0;
                        }
                    }
                } 
                return n_infer_batch;
            }

            std::vector<BoxArray> forward(std::vector<Frame> &frames, void *stream) override
            {   
                int n_batch = frames.size();  // return when batch size is less than 1.
                if (n_batch == 0)
                    return {};
                int n_infer_batch = get_n_infer_batch(n_batch);
                if (n_infer_batch == 0)
                    return {};
                // prepare cache.
                frame_prepare_cache(n_infer_batch);
                // preprocess.
                std::vector<AffineMatrix> affine_mats(n_infer_batch);
                size_t affine_mat_size = sizeof(affine_mats[0].d2i);
                // cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
                uint8_t *ptr_rgb = static_cast<uint8_t *>(preprocess_buffer_->device()) + n_infer_batch * upbound_<size_t>(affine_mat_size, 32);
                for (int i = 0; i < n_infer_batch; ++i)
                {
                    float *ptr_input = reinterpret_cast<float *>(input_buffer_->device()) + i * n_channels_ * height_ * width_;
                    float *ptr_affine_mat = reinterpret_cast<float *>(
                        reinterpret_cast<uint8_t *>(preprocess_buffer_->device()) + i * upbound_<size_t>(affine_mat_size, 32)
                    );
                    frame_preprocess(
                        frames[i], frame_buffer_, 
                        ptr_affine_mat, ptr_rgb, ptr_input, 
                        affine_mats[i], width_, height_, 
                        127.0f, 255.0f, stream
                    );
                }
                // foward...
                std::vector<void *> bindings = {input_buffer_->device(), bbox_predict_->device()};  // binding input and output.
                cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
                if (!infer_->forward(bindings, _stream))
                {
                    INFO_ERROR("Error happened when forward, check the dims of bindings.");
                    return {};
                }
                return postprocess(n_infer_batch, stream);
            }

            /// @brief Allocate memory only when allocated space is not enough.
            /// @param n_batch 
            void frame_prepare_cache(int n_batch)
            {
                // TODO: Do we need to specify the deivce_id.
                // (32B * n) + c * h * w * sizeof(uint8_t)
                if (preprocess_buffer_ == nullptr)
                    preprocess_buffer_.reset(new BaseMemory());
                preprocess_buffer_->device(upbound_<size_t>(sizeof(AffineMatrix().d2i), 32) * n_batch + height_ * width_ * n_channels_ * sizeof(uint8_t));

                // [n, c, h, w] float32 -> n * c * h * w * sizeof(float) bytes.
                size_t bytes_per_batch = height_ * width_ * n_channels_ * sizeof(float);
                if (input_buffer_ == nullptr)
                    input_buffer_.reset(new BaseMemory());  
                input_buffer_->device(n_batch * bytes_per_batch);
                
                // [n, 25200, n_classes + 5] float 32 -> (n * 25200 * n_classes + 5) bytes
                if (bbox_predict_ == nullptr)
                    bbox_predict_.reset(new BaseMemory());
                bbox_predict_->device(n_batch * bbox_dims_[1] * bbox_dims_[2] * sizeof(float));   
                
                // (32 + max_objtects * n_box_elements * sizeof(float)) * n_batch
                if (output_boxarray_ == nullptr)
                    output_boxarray_.reset(new BaseMemory());
                output_boxarray_->device((32 + max_objects * n_box_elements * sizeof(float)) * n_batch);  
                output_boxarray_->host((32 + max_objects * n_box_elements * sizeof(float)) * n_batch);
            }

            BoxArray forward(MixMat &mix_mat, void *stream = nullptr) override
            {
                std::vector<MixMat> mix_mats = {mix_mat};
                std::vector<BoxArray> ret = forward(mix_mats, stream);
                if (ret.empty())
                    return {};
                return ret.front();
            }

            std::vector<BoxArray> forward(std::vector<MixMat> &mix_mats, void *stream = nullptr) override
            {
                int n_batch = mix_mats.size();  // return when batch size is less than 1.
                if (n_batch == 0)
                    return {};

                int n_infer_batch = get_n_infer_batch(n_batch);
                if (n_infer_batch == 0)
                    return {};
                // prepare cache.
                mix_mat_prepare_cache(n_infer_batch);
                // preprocess.
                std::vector<AffineMatrix> affine_mats(n_infer_batch);
                size_t affine_mat_size = sizeof(affine_mats[0].d2i);
                for (int i = 0; i < n_infer_batch; ++i)
                {
                    float *ptr_input = reinterpret_cast<float *>(input_buffer_->device()) + i * n_channels_ * height_ * width_;
                    float *ptr_affine_mat = reinterpret_cast<float *>(
                        reinterpret_cast<uint8_t *>(preprocess_buffer_->device()) + i * upbound_<size_t>(affine_mat_size, 32)
                    );
                    mix_mat_preprocess(
                        mix_mats[i], ptr_affine_mat, mix_mats[i].device<uint8_t>(), 
                        ptr_input, affine_mats[i], width_, height_, 
                        127.0f, 255.0f, stream
                    );
                }
                // foward...
                std::vector<void *> bindings = {input_buffer_->device(), bbox_predict_->device()};  // binding input and output.
                cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
                if (!infer_->forward(bindings, _stream))
                {
                    INFO_ERROR("Error happened when forward, check the dims of bindings.");
                    return {};
                }
                auto ret = postprocess(n_infer_batch, stream);
                return ret;
            }   

            std::vector<BoxArray> postprocess(int n_infer_batch, void *stream = nullptr)
            {
                cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
                // postprocess
                for (int i = 0; i < n_infer_batch; ++i)
                {
                    float *bbox_predict = static_cast<float *>(bbox_predict_->device());
                    int n_bboxes = bbox_dims_[1];
                    int n_classes = n_classes_;
                    int n_bbox_info = bbox_dims_[2];
                    float *d2i = reinterpret_cast<float *>(
                        reinterpret_cast<uint8_t *>(preprocess_buffer_->device()) + i * upbound_<size_t>(sizeof(AffineMatrix::d2i), 32)
                    );
                    float *bbox_out = reinterpret_cast<float *>(
                        reinterpret_cast<uint8_t *>(output_boxarray_->device()) + 32 + i * max_objects * n_box_elements * sizeof(float)
                    );
                    CHECK_CUDA_RUNTIME(cudaMemsetAsync(bbox_out, 0, sizeof(float), _stream));
                    tinycv::yolo::postprocess(
                        bbox_predict, n_bboxes, n_classes, n_bbox_info, 
                        d2i, bbox_out, max_objects, confidence_threshold_, 
                        nms_threshold_, stream
                    );
                }
                CHECK_CUDA_RUNTIME(cudaMemcpyAsync(output_boxarray_->host(), output_boxarray_->device(), output_boxarray_->device_data_size(), cudaMemcpyDeviceToHost, _stream));
                CHECK_CUDA_RUNTIME(cudaStreamSynchronize(_stream));
                std::vector<BoxArray> ret(n_infer_batch);
                // fetch the nms result.
                for (int i = 0; i < n_infer_batch; ++i)
                {
                    float *ptr_bboxes = reinterpret_cast<float *>(
                        reinterpret_cast<uint8_t *>(output_boxarray_->host()) + 32 + i * max_objects * n_box_elements * sizeof(float)
                    );
                    int cnt = std::min(max_objects, static_cast<int>(*ptr_bboxes));
                    BoxArray &box_arr = ret[i];
                    box_arr.reserve(cnt);
                    for (int j = 0; j < cnt; ++j)
                    {
                        float *ptr_bbox = reinterpret_cast<float *>(
                            reinterpret_cast<uint8_t *>(ptr_bboxes) + 32 + j * n_box_elements * sizeof(float)
                        );
                        int keep_flag = ptr_bbox[6];
                        if (!keep_flag)
                            continue;
                        int label = ptr_bbox[5];
                        box_arr.emplace_back(ptr_bbox[0], ptr_bbox[1], ptr_bbox[2], ptr_bbox[3], ptr_bbox[4], label);
                    }
                }
                return ret;    
            }

            /// @brief Allocate memory only when allocated space is not enough.
            /// @param n_batch 
            void mix_mat_prepare_cache(int n_batch)
            {
                // TODO: Do we need to specify the deivce_id.
                // (32B * n)
                if (preprocess_buffer_ == nullptr)
                    preprocess_buffer_.reset(new BaseMemory());
                preprocess_buffer_->device(upbound_<size_t>(sizeof(AffineMatrix().d2i), 32) * n_batch);

                // [n, c, h, w] float32 -> n * c * h * w * sizeof(float) bytes.
                size_t bytes_per_batch = height_ * width_ * n_channels_ * sizeof(float);
                if (input_buffer_ == nullptr)
                    input_buffer_.reset(new BaseMemory());  
                input_buffer_->device(n_batch * bytes_per_batch);
                
                // [n, 25200, n_classes + 5] float 32 -> (n * 25200 * n_classes + 5) bytes
                if (bbox_predict_ == nullptr)
                    bbox_predict_.reset(new BaseMemory());
                bbox_predict_->device(n_batch * bbox_dims_[1] * bbox_dims_[2] * sizeof(float));   
                
                // (32 + max_objtects * n_box_elements * sizeof(float)) * n_batch
                if (output_boxarray_ == nullptr)
                    output_boxarray_.reset(new BaseMemory());
                output_boxarray_->device((32 + max_objects * n_box_elements * sizeof(float)) * n_batch);  
                output_boxarray_->host((32 + max_objects * n_box_elements * sizeof(float)) * n_batch);
            }


        private:
            std::shared_ptr<trt::TRTInfer> infer_;
            std::string engine_file_;
            float confidence_threshold_ = 0.5;
            float nms_threshold_ = 0.5;

            // Use BaseMemory here cause the data transmission from GPU to CPU rely on stream heavily.
            std::shared_ptr<BaseMemory> frame_buffer_ = nullptr;  // Read from video or image, it should be uint8_t type.
            std::shared_ptr<BaseMemory> preprocess_buffer_ = nullptr;  // 32B(affine matrix) * n + (c * h * w) B.
            std::shared_ptr<BaseMemory> input_buffer_ = nullptr;  // The network's input, which may be [n, 3, 640, 640], it should be float32.
            std::shared_ptr<BaseMemory> bbox_predict_ = nullptr;  // The network's output, which may be[n, 25200, 85], it should be float32.
            std::shared_ptr<BaseMemory> output_boxarray_ = nullptr;  // to handle the output and contains the information, it should be float32. 
            std::shared_ptr<BaseMemory> visual_buffer_ = nullptr; 
            int n_classes_ = -1;
            int height_ = -1, width_ = -1, n_channels_ = -1;
            bool is_dynamic_mode_ = false; 

            std::vector<int> bbox_dims_;
            const int max_objects = 1024;  // 1024 objects limit 
            const int n_box_elements = 8;  // for each box: left, right, top, bottom, confidence, class, keep_flag, row_idx
        };

        std::shared_ptr<YoloInfer> load(const std::string &engine_file, float confidence_threshold, float nms_threshold)
        {
            YoloInferImpl *instance = new YoloInferImpl();
            if (!instance->load(engine_file, confidence_threshold, nms_threshold))
            {
                delete instance;
                instance = nullptr;
            }
            return std::shared_ptr<YoloInfer>(instance);
        }
    }
}