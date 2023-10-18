#include "host_process.hpp"
#include <fstream>
#include <vector>
#include <cstdio>
using namespace std;

namespace host_process
{
    cv::Mat cpuWarpAffine(const cv::Mat &src_image, float *trans_mat, const ImageSize &dst_size)
    {
        cv::Mat i2d(2, 3, CV_32F, trans_mat);
        cv::Mat dst_image(dst_size.height, dst_size.width, CV_8UC3);
        cv::warpAffine(
            src_image, 
            dst_image, 
            i2d, 
            dst_image.size(), 
            cv::INTER_LINEAR, 
            cv::BORDER_CONSTANT, 
            cv::Scalar::all(114)
        );
        return dst_image;
    }

    void cpucvBGR2RGB(float *ptr_rgb, const cv::Mat &bgr_image)
    {
        int n_rows = bgr_image.rows, n_cols = bgr_image.cols;
        int image_area = n_rows * n_cols;
        float *ptr_r = ptr_rgb;
        float *ptr_g = ptr_rgb + image_area;
        float *ptr_b = ptr_rgb + image_area * 2;
        uint8_t *ptr_bgr = bgr_image.data;
        for (int i = 0; i < image_area; ++i, ptr_bgr += 3)
        {
            *ptr_b++ = *ptr_bgr / 255.0f; 
            *ptr_g++ = *(ptr_bgr + 1) / 255.0f; 
            *ptr_r++ = *(ptr_bgr + 2) / 255.0f; 
        }
    }
};


