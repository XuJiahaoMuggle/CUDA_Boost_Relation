#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <host_process/host_process.hpp>
#include <yolo/preprocess.hpp>
#include <utils/utils.h>
#include <string>
#include <ctime>

void preprocess_benchmark(int iteration=20)
{
    // prepare
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    const std::string image_path = "./train.jpg";
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
        return;
    int height = image.rows;
    int width = image.cols;
    int n_channels = image.channels();
    size_t bytes = height * width * n_channels;
    tinycv::yolo::AffineMatrix affine;
    affine.compute(std::make_tuple(width, height), std::make_tuple(640, 640));
    float *host_trans = affine.i2d;
    float *host_dst = new float[bytes];
    uint8_t *host_src = image.data;

    float *device_trans = nullptr;
    float *device_dst = nullptr;
    uint8_t *device_src = nullptr;
    CHECK_CUDA_RUNTIME(cudaMalloc(&device_trans, sizeof(affine.d2i)));
    CHECK_CUDA_RUNTIME(cudaMemcpy(device_trans, affine.d2i, sizeof(affine.d2i), cudaMemcpyHostToDevice));
    CHECK_CUDA_RUNTIME(cudaMalloc(&device_dst, bytes * sizeof(float)));
    CHECK_CUDA_RUNTIME(cudaMalloc(&device_src, bytes));
    CHECK_CUDA_RUNTIME(cudaMemcpy(device_src, host_src, bytes, cudaMemcpyHostToDevice));
    std::vector<std::pair<float, float>> elapsed_times;
    elapsed_times.reserve(iteration);
    cudaEvent_t device_begin, device_end;
    CHECK_CUDA_RUNTIME(cudaEventCreate(&device_begin));
    CHECK_CUDA_RUNTIME(cudaEventCreate(&device_end));
    while (iteration--)
    {
        CHECK_CUDA_RUNTIME(cudaEventRecord(device_begin));
        tinycv::yolo::warp_affine_bilinear_normalize(
            device_src, width, height, 
            device_dst, 640, 640, 
            device_trans  
        );
        CHECK_CUDA_RUNTIME(cudaEventRecord(device_end));
        CHECK_CUDA_RUNTIME(cudaEventSynchronize(device_end));
        float device_elapesd;
        CHECK_CUDA_RUNTIME(cudaEventElapsedTime(&device_elapesd, device_begin, device_end));
        CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());
        clock_t host_stamp = clock();
        auto warped_image = host_process::cpuWarpAffine(
            image, host_trans, {640, 640}
        );
        host_process::cpucvBGR2RGB(host_dst, warped_image);
        clock_t end_stamp = clock();
        float host_elapesd = static_cast<float>(end_stamp - host_stamp) / (CLOCKS_PER_SEC / 1000);
        elapsed_times.emplace_back(device_elapesd, host_elapesd);
    }
    float device_elapsed = 0;
    float host_elapsed = 0;
    for (int i = 0; i < elapsed_times.size(); ++i)
    {
        device_elapsed += elapsed_times[i].first;
        host_elapsed += elapsed_times[i].second;
        INFO("Iteration %d's elapsed time of device: %.2fms, host: %.2fms", i, elapsed_times[i].first, elapsed_times[i].second);
        elapsed_times[i].first;
    }
    device_elapsed /= elapsed_times.size();
    host_elapsed /= elapsed_times.size();
    INFO("Average elapsed time of device: %.2fms, host: %.2fms", device_elapsed, host_elapsed);
}

int main()
{
    preprocess_benchmark();
    return 0;


}

