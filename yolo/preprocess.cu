#include "preprocess.hpp"
#include <utils/utils.h>
#include <tuple>
#include <image_proc/cvt_color.cuh>
#include <algorithm>
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#ifdef __cplusplus
}
#endif

#define BLOCK_SIZE 32 
#define CLIP(VAL, LOW, HIGH) (VAL) < (LOW) ? (LOW) : ((VAL) > (HIGH) ? (HIGH) : (VAL))
#define CLIP_TO_UINT8(VAL) (uint8_t)(CLIP((VAL), (0.0f), (255.0f)))

template <typename ty>
static inline ty upbound_(ty val, ty n)
{
    return (val - 1 + n) / n * n;
}

/// @brief For CHW format strides: [HW, W, 1], for HWC format strides: [WC, C, 1]
static __global__ void warp_affine_bilinear_normalize_kernel(
    uint8_t *ptr_src, int src_width, int src_height,
    float *ptr_dst, int dst_width, int dst_height,
    float *ptr_affine_mat, float val, float coef
){
    uint32_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_idx >= dst_width || y_idx >= dst_height)
        return;
    float mat_x0 = ptr_affine_mat[0];
    float mat_y0 = ptr_affine_mat[1];
    float mat_z0 = ptr_affine_mat[2];
    float mat_x1 = ptr_affine_mat[3];
    float mat_y1 = ptr_affine_mat[4];
    float mat_z1 = ptr_affine_mat[5];
    // project
    float src_x = mat_x0 * x_idx + mat_y0 * y_idx + mat_z0;
    float src_y = mat_x1 * x_idx + mat_y1 * y_idx + mat_z1;
    float r_val, g_val, b_val;
    // out of range
    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
    {
        r_val = val;
        g_val = val;
        b_val = val;
    } 
    else
    {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = y_high - src_y;
        float hx = x_high - src_x;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t const_val[3] = {(uint8_t)val, (uint8_t)val, (uint8_t)val};  // R G B
        uint8_t *src_v1 = const_val;  // [y_low, x_low]
        uint8_t *src_v2 = const_val;  // [y_low, x_high]
        uint8_t *src_v3 = const_val;  // [y_high, x_low]
        uint8_t *src_v4 = const_val;  // [y_high, x_high]
        // for HWC format, stride: [WC, C, 1]
        if (y_low >= 0)
        {
            if (x_low >= 0)
                src_v1 = ptr_src + y_low * src_width * 3 + x_low * 3;
            if (x_high < src_width)
                src_v2 = ptr_src + y_low * src_width * 3 + x_high * 3;    
        }

        if (y_high < src_height)
        {
            if (x_low >= 0)
                src_v3 = ptr_src + y_high * src_width * 3 + x_low * 3;
            if (x_high < src_width)
                src_v4 = ptr_src + y_high * src_width * 3 + x_high * 3;
        }
        r_val = floorf(w1 * src_v1[0] + w2 * src_v2[0] + w3 * src_v3[0] + w4 * src_v4[0] + 0.5f);
        g_val = floorf(w1 * src_v1[1] + w2 * src_v2[1] + w3 * src_v3[1] + w4 * src_v4[1] + 0.5f);
        b_val = floorf(w1 * src_v1[2] + w2 * src_v2[2] + w3 * src_v3[1] + w4 * src_v4[2] + 0.5f);
    }
    // for CHW format stride: [HW, W, 1]
    size_t dst_idx = y_idx * dst_width + x_idx;
    size_t area = dst_height * dst_width;
    float *ptr_dst_r = ptr_dst + dst_idx;
    float *ptr_dst_g = ptr_dst_r + area;
    float *ptr_dst_b = ptr_dst_g + area; 
    *ptr_dst_r = CLIP(r_val, 0.0, 255.0) / coef; 
    *ptr_dst_g = CLIP(g_val, 0.0, 255.0) / coef;
    *ptr_dst_b = CLIP(b_val, 0.0, 255.0) / coef;
}

namespace tinycv
{
    namespace yolo
    {
        void warp_affine_bilinear_normalize(
            uint8_t *ptr_src, int src_width, int src_height,
            float *ptr_dst, int dst_width, int dst_height,
            float *ptr_affine_mat, float val, float cof,
            void *stream
        ){
            cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
            dim3 block_dim{BLOCK_SIZE, BLOCK_SIZE};
            dim3 grid_dim{
                (static_cast<uint32_t>(dst_width) + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (static_cast<uint32_t>(dst_height) + BLOCK_SIZE - 1) / BLOCK_SIZE
            };
            CHECK_CUDA_KERNEL(
                warp_affine_bilinear_normalize_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_src, src_width, src_height, 
                    ptr_dst, dst_width, dst_height, 
                    ptr_affine_mat, val, cof
                )
            );    
        }

        void frame_preprocess(
            const Frame &frame, 
            std::shared_ptr<BaseMemory> &frame_buffer,  // uint8
            float *ptr_affine_mat,  // 32B + 3 * height * width uin8_t
            uint8_t *ptr_rgb,
            float *ptr_input,  // float32
            AffineMatrix &affine_mat,
            int network_width, 
            int network_height,
            float val, 
            float coef,
            void *stream, 
            bool sync
        ){
            int width = frame.width();
            int height = frame.height();
            int pix_fmt = frame.pixel_format();
            MemoryType mem_type = frame.memory_type();
            size_t bytes = frame.bytes();
            if (frame_buffer == nullptr)
                frame_buffer.reset(new BaseMemory());
            frame_buffer->device(bytes);
            assert(mem_type != MemoryType::MEM_UNKNOWN);
            // convert data to GPU
            cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
            if (mem_type == MemoryType::MEM_UNREGISTED || mem_type == MemoryType::MEM_PAGE_LOCKED)
                CHECK_CUDA_RUNTIME(
                    cudaMemcpyAsync(frame_buffer->device(), frame.data(), bytes, cudaMemcpyHostToDevice, stream_)
                );
            else // (frame.memory_type() == MemoryType::MEM_DEVICE)
                CHECK_CUDA_RUNTIME(
                    cudaMemcpyAsync(frame_buffer->device(), frame.data(), bytes, cudaMemcpyDeviceToDevice, stream_)
                );
            // convert color
            uint8_t *ptr_yuv = static_cast<uint8_t *>(frame_buffer->device());
            convert_yuv_to_rgb_hwc(ptr_yuv, pix_fmt, ptr_rgb, width, height, stream);

            // warp affine
            affine_mat.compute(std::make_tuple(width, height), std::make_tuple(network_width, network_height));
            float *ptr_affine_mat_host = affine_mat.d2i;
            CHECK_CUDA_RUNTIME(cudaMemcpyAsync(ptr_affine_mat, ptr_affine_mat_host, sizeof(affine_mat.d2i), cudaMemcpyHostToDevice, stream_));
            warp_affine_bilinear_normalize(
                ptr_rgb, width, height,
                ptr_input, network_width, network_height,
                ptr_affine_mat, val, coef, stream
            );
            if (sync)
                CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream_));
        }

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
            void *stream, 
            bool sync
        ){
            int width = mix_mat.dims()[1];
            int height = mix_mat.dims()[0];
            cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
            // warp affine
            affine_mat.compute(std::make_tuple(width, height), std::make_tuple(network_width, network_height));
            float *ptr_affine_mat_host = affine_mat.d2i;
            CHECK_CUDA_RUNTIME(cudaMemcpyAsync(ptr_affine_mat, ptr_affine_mat_host, sizeof(affine_mat.d2i), cudaMemcpyHostToDevice, stream_));
            warp_affine_bilinear_normalize(
                ptr_rgb, width, height,
                ptr_input, network_width, network_height,
                ptr_affine_mat, val, coef, stream
            );
            if (sync)
                CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream_));
        }
    }
}
