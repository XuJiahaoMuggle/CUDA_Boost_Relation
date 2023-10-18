#include <cpm/cpm.hpp>
#include <yolo/yolo_infer.hpp>
#include <mix_memory/mix_mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <utils/utils.h>
#include <video_proc/video_proc.h>
#include <cuda_runtime.h>

#include <ctime>
#include <iostream>
#include <vector>
#include <string>
#include <thread>

#ifdef __cplusplus
extern "C"
{
#endif

#include <libavcodec/avcodec.h>

#ifdef __cplusplus
}
#endif
using cpm_ty = tinycv::Cpm<tinycv::yolo::BoxArray, tinycv::MixMat, tinycv::yolo::YoloInfer>;

void benchmark_one_thread(const std::string &video_path, const std::string engine_path, int n)
{
    tinycv::VideoCap video_cap(video_path, "");
    std::vector<std::shared_ptr<tinycv::yolo::YoloInfer>> infers(n, nullptr);
    std::vector<cudaStream_t> streams(n);
    for (int i = 0; i < n; ++i)
    {
        CHECK_CUDA_RUNTIME(cudaStreamCreate(&streams[i]));
        infers[i] = tinycv::yolo::load("yolov5s.engine");
    }
    tinycv::MixMat mix_mat;    
    cv::Mat mat;
    bool done = false;;
    cudaEvent_t begin, end;
    CHECK_CUDA_RUNTIME(cudaEventCreate(&begin));
    CHECK_CUDA_RUNTIME(cudaEventCreate(&end));
    float cost = 0.;
    CHECK_CUDA_RUNTIME(cudaEventRecord(begin, streams[0]));
    while (!done)
    {
        mix_mat = video_cap.read_mix_mat(streams[0]);
        for (int i = 0; i < n; ++i)
        {
            mix_mat.set_stream(streams[i]);
            if (mix_mat.empty())
            {
                done = true;
                break;
            }
            auto res = infers[i]->forward(mix_mat, streams[i]);
            mat = cv::Mat(video_cap.get_height(), video_cap.get_width(), CV_8UC3, mix_mat.host());
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
        }    
        cv::imshow("infer_video", mat);
        cv::waitKey(1);
    }
    CHECK_CUDA_RUNTIME(cudaEventRecord(end, streams[n - 1]));
    CHECK_CUDA_RUNTIME(cudaEventSynchronize(end));
    CHECK_CUDA_RUNTIME(cudaEventElapsedTime(&cost, begin, end));
    std::cout << "one_thread " << n << " cost: " << cost << " ms" << std::endl;
    CHECK_CUDA_RUNTIME(cudaEventDestroy(begin));
    CHECK_CUDA_RUNTIME(cudaEventDestroy(end));
    for (int i = 0; i < n; ++i)
        CHECK_CUDA_RUNTIME(cudaStreamSynchronize(streams[i]));
}

void benchmark_cpm(const std::string &video_path, const std::string engine_path, int n)
{
    std::vector<cpm_ty> cpms(n);
    std::vector<cudaStream_t> streams(n, nullptr);
    bool status = true;
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
    tinycv::VideoCap video_cap(video_path, "");
    int height = video_cap.get_height();
    int width = video_cap.get_width();
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
        in_mat = video_cap.read_mix_mat(streams[0]);
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

int main(int argc, char **argv)
{   
    const std::string video_path = "./fall_video.mp4";
    const std::string engine_path = "./yolov5s.engine";
    av_log_set_level(AV_LOG_QUIET);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    int n_models = 3;
    benchmark_cpm(video_path, engine_path, n_models);
    // benchmark_one_thread(video_path, engine_path, n_models);
    return 0;
}