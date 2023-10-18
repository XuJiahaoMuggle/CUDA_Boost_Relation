#include <video_proc/video_proc.h>
#include <yolo/yolo_infer.hpp>
#include <cpm/cpm.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <vector>

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    // const std::string url = "video=HP Wide Vision HD Camera"; 
    // const std::string input_fmt_name = "dshow";  
    const std::string url = "./fall_video.mp4"; 
    const std::string input_fmt_name = "";  
    tinycv::NVVideoCap nv_cap(url, input_fmt_name, 0, true, false);
    // initialize cpm
    using cpm_ty = tinycv::Cpm<tinycv::yolo::BoxArray, tinycv::MixMat, tinycv::yolo::YoloInfer>;
    const int n = 3;
    std::vector<cpm_ty> cpms(n);
    std::vector<cudaStream_t> streams(n, nullptr);
    bool status = true;
    const std::string engine_path = "../../quant/yolov5/quant_yolov5n_replace_to_quantization.engine";  // "yolov5s.engine";
    for (int i = 0; i < n; ++i)
    {
        CHECK_CUDA_RUNTIME(cudaStreamCreate(&streams[i]));
        status &= cpms[i].start(
            [&engine_path]() { return tinycv::yolo::load(engine_path); },
            1,
            streams[i]
        );
    }
    if (!status)
        INFO_ERROR("CPM launch failed!\n");

    int height = nv_cap.get_height();
    int width = nv_cap.get_width();
    tinycv::MixMat in_mat({height, width, 3}, tinycv::DataType::UINT8);
    cv::Mat image;
    bool done = false;
    cudaEvent_t begin, end;
    CHECK_CUDA_RUNTIME(cudaEventCreate(&begin));
    CHECK_CUDA_RUNTIME(cudaEventCreate(&end));
    float cost = 0.;
    CHECK_CUDA_RUNTIME(cudaEventRecord(begin, streams[0]));
    std::vector<std::shared_future<tinycv::yolo::BoxArray>> futs(n);
    while (!done)
    {
        in_mat = nv_cap.read_mix_mat(streams[0]);
        for (int i = 0; i < n; ++i)
        {
            in_mat.set_stream(streams[i]);
            if (in_mat.empty())
            {
                done = true;
                break;
            }
            futs[i] = cpms[i].commit(in_mat);
        }
        for (int i = 0; i < n && !done; ++i)
        {
            image = cv::Mat(height, width, CV_8UC3, in_mat.host());
            auto boxes = futs[i].get();
            for (auto &box: boxes)
            {
                uint8_t b = 0, g = 0, r = 0;
                std::tie(b, g, r) = tinycv::yolo::random_color(box.class_label);
                cv::rectangle(
                    image,
                    cv::Point(box.left, box.top),
                    cv::Point(box.right, box.bottom),
                    cv::Scalar(b, g, r),
                    5
                );
            }
        }
        cv::imshow("cpm", image);
        cv::waitKey(1);
    }
    CHECK_CUDA_RUNTIME(cudaEventRecord(end, streams[n - 1]));
    CHECK_CUDA_RUNTIME(cudaEventSynchronize(end));
    CHECK_CUDA_RUNTIME(cudaEventElapsedTime(&cost, begin, end));
    std::cout << "cpm " << n << " cost: " << cost << " ms" << std::endl;
    CHECK_CUDA_RUNTIME(cudaEventDestroy(begin));
    CHECK_CUDA_RUNTIME(cudaEventDestroy(end));
    for (int i = 0; i < n; ++i)
        CHECK_CUDA_RUNTIME(cudaStreamSynchronize(streams[i]));
}