#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <yolo/yolo_infer.hpp>
#include <utils/utils.h>
#include <ctime>

using infer_ty = std::shared_ptr<tinycv::yolo::YoloInfer>;
using tensor_ty = tinycv::MixMat;

void warm_up(infer_ty &infer, const int iteration = 20) 
{
    INFO("Start warm up for %d iterations.", iteration);
    if (infer == nullptr)
    {
        INFO_ERROR("Invalid infer detected when warm up.");
        return ;
    }
    int height = 1080;
    int cols = 1920;
    int n_channels = 3;
    size_t n_bytes = height * cols * n_channels;
    tensor_ty tensor({height, cols, n_channels}, tinycv::UINT8);
    clock_t begin = clock();
    for (int i = 0; i < iteration; ++i)
        infer->forward(tensor);
    clock_t end = clock();
    float elapsed_time = static_cast<float>(end - begin) / static_cast<float>(CLOCKS_PER_SEC / 1000);
    INFO("Warm up info: total time cost: %.2f ms, average time cost: %.2f ms", elapsed_time, elapsed_time / iteration);
}

int main(int argc, char **argv)
{
    std::string image_path = "./train.jpg";
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    cv::Mat image = cv::imread(image_path);
    
    std::string engine_path = "../../quant/yolov5/engine_files/yolov5n-ptq-percentile-99.99-1024.engine";
    auto infer = tinycv::yolo::load(engine_path);
    warm_up(infer);

    tinycv::MixMat in_mat({image.rows, image.cols, image.channels()}, tinycv::DataType::UINT8);
    int height = image.rows, width = image.cols, n_channels = image.channels();
    size_t n_bytes = static_cast<size_t>(height) * width *n_channels; 
    in_mat.ref_data({image.rows, image.cols, image.channels()}, image.data, n_bytes, nullptr, 0);
    INFO("Start inference...");
    auto boxes = infer->forward(in_mat);
    for (auto &each: boxes)
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = tinycv::yolo::random_color(each.class_label);
        cv::rectangle(
            image, 
            cv::Point(each.left, each.top),
            cv::Point(each.right, each.bottom),
            cv::Scalar(b, g, r),
            5
        );
    }
    cv::imshow("infer_image", image);
    cv::waitKey();
    return 0;
}