#ifndef HOST_PROCESS_HPP
#define HOST_PROCESS_HPP
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace host_process{
    /**
     * @brief Contains the image's height and width.
     * @note The format is as opencv format: width, height.
     */
    struct ImageSize
    {
        uint32_t width, height;
        ImageSize(uint32_t width, uint32_t height) : width(width), height(height) {}
        ImageSize() : width(0), height(0){}
    };

    cv::Mat cpuWarpAffine(
        const cv::Mat &src_image, 
        float *trans_mat,
        const ImageSize &dst_size
    );

    void cpucvBGR2RGB(float *ptr_rgb, const cv::Mat &bgr_image);
};

#endif  // HOST_PROCESS_HPP