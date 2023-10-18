#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <yolo/yolo_infer.hpp>
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#ifdef __cplusplus
}
#endif
int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    av_log_set_level(AV_LOG_QUIET);
    const std::string url = "video=HP Wide Vision HD Camera"; 
    const std::string input_fmt_name = "dshow";  
    tinycv::VideoCap video_cap;
    video_cap.open(url, input_fmt_name);
    const std::string engine_name = "../../quant/yolov5/engine_files/yolov5n-percentile-99.9999-1024.engine";
    // const std::string engine_name = "../../quant/yolov5/engine_files/yolov5n.engine";
    auto infer = tinycv::yolo::load(engine_name, 0.4, 0.4);
    Frame frame;
    tinycv::MixMat mix_mat;
    int cnt = 0;
    int n_total_frames = video_cap.get_n_frames();
    int n_decoded_frames = 0;
    clock_t begin = clock();
    clock_t new_begin = begin;
    cudaStream_t infer_stream;
    CHECK_CUDA_RUNTIME(cudaStreamCreate(&infer_stream));
    while (true)
    {
        mix_mat = video_cap.read_mix_mat(infer_stream);
        if (mix_mat.empty())
            break;
        auto res = infer->forward(mix_mat, infer_stream);
        cv::Mat mat(video_cap.get_height(), video_cap.get_width(), CV_8UC3, mix_mat.host());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        for (auto &each: res)
        {
            uint8_t b, g, r;
            std::tie(b, g, r) = tinycv::yolo::random_color(each.class_label);
            cv::rectangle(
                mat, cv::Point(each.left, each.top), 
                cv::Point(each.right, each.bottom), 
                cv::Scalar(b, g, r), 5
            );
        }
        cv::imshow("infer_video", mat);
        cv::waitKey(1);
        ++n_decoded_frames;
        ++cnt;
        if (cnt == 100)
        {
            std::cout << 100 / (double)(clock() - new_begin) * (CLOCKS_PER_SEC) << std::endl;
            cnt = 0;
            new_begin = clock();
        }
    }
    INFO("Decocd %d / %d frames", n_decoded_frames, n_total_frames);
    INFO("FPS %f", n_decoded_frames / (double)(clock() - begin) * (CLOCKS_PER_SEC));
    CHECK_CUDA_RUNTIME(cudaStreamSynchronize(infer_stream));
    CHECK_CUDA_RUNTIME(cudaStreamDestroy(infer_stream));
}