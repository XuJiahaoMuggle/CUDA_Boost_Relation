#ifndef PREPROCESS_CUH
#define PREPROCESS_CUH

#include <tuple>
#include <algorithm>
#include <memory>
#include <mix_memory/base_memory.hpp>
#include <mix_memory/mix_mat.hpp>
#include <video_proc/video_proc.h>

namespace tinycv
{
    namespace yolo
    {
        struct AffineMatrix 
        {
            float i2d[6];  // image to dst(network), 2x3 matrix
            float d2i[6];  // dst to image, 2x3 matrix
            void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to)
            {
                float scale_x = std::get<0>(to) / (float)std::get<0>(from);
                float scale_y = std::get<1>(to) / (float)std::get<1>(from);
                float scale = std::min(scale_x, scale_y);
                i2d[0] = scale;
                i2d[1] = 0;
                i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 + scale * 0.5 - 0.5;
                i2d[3] = 0;
                i2d[4] = scale;
                i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 + scale * 0.5 - 0.5;

                double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
                D = D != 0. ? double(1.) / D : double(0.);
                double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
                double b1 = -A11 * i2d[2] - A12 * i2d[5];
                double b2 = -A21 * i2d[2] - A22 * i2d[5];

                d2i[0] = A11;
                d2i[1] = A12;
                d2i[2] = b1;
                d2i[3] = A21;
                d2i[4] = A22;
                d2i[5] = b2;
            }
        };        

        void warp_affine_bilinear_normalize(
            uint8_t *ptr_src,
            int src_width,
            int src_height,
            float *ptr_dst,
            int dst_width,
            int dst_height,
            float *ptr_affine_mat,
            float val = 127.0,
            float cof = 255.0,
            void *stream = nullptr
        );

        void frame_preprocess(
            const Frame &frame, 
            std::shared_ptr<BaseMemory> &frame_buffer,  // uint8
            float *ptr_affine_mat,  // 32B + 3 * height * width uin8_t
            uint8_t *ptr_rgb,
            float *ptr_input,  // float32
            AffineMatrix &affine_mat,
            int network_width, 
            int network_height,
            float val = 127, 
            float coef = 255.0,
            void *stream = nullptr, 
            bool sync = false
        );

        void mix_mat_preprocess(
            MixMat &mix_mat,
            float *ptr_affine_mat,  // 32B + 3 * height * width uin8_t
            uint8_t *ptr_rgb,
            float *ptr_input,  // float32
            AffineMatrix &affine_mat,
            int network_width, 
            int network_height,
            float val, 
            float coef,
            void *stream = nullptr, 
            bool sync = false
        );
    }
}



#endif  // PREPROCESS_CUH