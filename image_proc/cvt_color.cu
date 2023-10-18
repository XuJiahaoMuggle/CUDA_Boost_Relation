#include "cvt_color.cuh"
#include <cassert>
#include <vector>
#include <utils/utils.h>

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

namespace tinycv
{
    // y u v range [0, 255]
    __device__ __inline__ void yuvj_map_to_rgb(uint8_t *r, uint8_t *g, uint8_t *b, float y, float u , float v)
    {
        *r = CLIP_TO_UINT8(y + 1.402 * (v - 128));  // R
        *g = CLIP_TO_UINT8(y - 0.34413 * (u - 128) - 0.71414 * (v - 128));  // G
        *b = CLIP_TO_UINT8(y + 1.772 * (u - 128));  // B
    }

    // y range [16, 235] u b range [16, 239]
    __device__ __inline__ void yuv_map_to_rgb(uint8_t *r, uint8_t *g, uint8_t *b, float y, float u , float v)
    {
        *r = CLIP_TO_UINT8(1.164 * (y - 16) + 2.018 * (u - 128));
        *g = CLIP_TO_UINT8(1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128));
        *b = CLIP_TO_UINT8(1.164 * (y - 16) + 1.596 * (v - 128));
    }

    __global__ void convert_nv12_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    ) {
        uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;
        uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (y_idx >= height || x_idx >= width)
            return;
        // strides [width * c, c, 1], idx [y_idx, x_idx, 0 / 1 / 2]
        uint8_t *rgb_ptr =  dst_data + y_idx * width * 3 + x_idx * 3;

        // strides [width, 1]
        float y_val = *(src_data + y_idx * width + x_idx);
        size_t offset = height * width;
        float u_val = *(src_data + offset + y_idx / 2 * width + x_idx / 2 * 2); 
        float v_val = *(src_data + offset + y_idx / 2 * width + x_idx / 2 * 2 + 1);
        // ?
        yuv_map_to_rgb(rgb_ptr + 2, rgb_ptr + 1, rgb_ptr + 0, y_val, u_val, v_val);
    }

    /**
     * @brief A series of 420p convert function, include yuv420p, yuvj420p, support hwc result.
     */
    __global__ void convert_yuv420p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    ){
        uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;
        uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (y_idx >= height || x_idx >= width)
            return;

        // strides [width * c, c, 1], idx [y_idx, x_idx, 0 / 1 / 2]
        uint8_t *rgb_ptr =  dst_data + y_idx * width * 3 + x_idx * 3;

        // strides [width, 1]
        float y_val = *(src_data + y_idx * width + x_idx);
        // strides [width / 2, 1]
        float u_val = *(src_data + height * width + y_idx * width / 4 + x_idx / 2);
        // strides [width / 2, 1]
        float v_val = *(src_data + height * width + height * width / 4 + y_idx * width / 4 + x_idx / 2);

        yuv_map_to_rgb(rgb_ptr, rgb_ptr + 1, rgb_ptr + 2, y_val, u_val, v_val);
    }

    __global__ void convert_yuvj420p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    ){
        uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;
        uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (y_idx >= height || x_idx >= width)
            return;

        // strides [width * c, c, 1], idx [y_idx, x_idx, 0 / 1 / 2]
        uint8_t *rgb_ptr =  dst_data + y_idx * width * 3 + x_idx * 3;

        // strides [width, 1]
        float y_val = *(src_data + y_idx * width + x_idx);
        // strides [width / 2, 1]
        float u_val = *(src_data + height * width + y_idx * width / 4 + x_idx / 2);
        // strides [width / 2, 1]
        float v_val = *(src_data + height * width + height * width / 4 + y_idx * width / 4 + x_idx / 2);

        yuvj_map_to_rgb(rgb_ptr, rgb_ptr + 1, rgb_ptr + 2, y_val, u_val, v_val);
    }

    /**
     * @brief A series of 422p convert function, include yuv422p, yuvj422p, support hwc result.
     */
    __global__ void convert_yuvj422p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    ){
        uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;
        uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (y_idx >= height || x_idx >= width)
            return;

        // strides [width * c, c, 1], idx [y_idx, x_idx, 0 / 1 / 2]
        uint8_t *rgb_ptr =  dst_data + y_idx * width * 3 + x_idx * 3;

        // strides [width, 1]
        float y_val = *(src_data + y_idx * width + x_idx);
        // strides [width / 2, 1]
        float u_val = *(src_data + height * width + y_idx * width / 2 + x_idx / 2);
        // strides [width / 2, 1]
        float v_val = *(src_data + height * width + height * width / 2 + y_idx * width / 2 + x_idx / 2);

        yuvj_map_to_rgb(rgb_ptr, rgb_ptr + 1, rgb_ptr + 2, y_val, u_val, v_val);
    }

    __global__ void convert_yuv422p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    ){
        uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;
        uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (y_idx >= height || x_idx >= width)
            return;

        // strides [width * c, c, 1], idx [y_idx, x_idx, 0 / 1 / 2]
        uint8_t *rgb_ptr =  dst_data + y_idx * width * 3 + x_idx * 3;

        // strides [width, 1]
        float y_val = *(src_data + y_idx * width + x_idx);
        // strides [width / 2, 1]
        float u_val = *(src_data + height * width + y_idx * width / 2 + x_idx / 2);
        // strides [width / 2, 1]
        float v_val = *(src_data + height * width + height * width / 2 + y_idx * width / 2 + x_idx / 2);

        yuv_map_to_rgb(rgb_ptr, rgb_ptr + 1, rgb_ptr + 2, y_val, u_val, v_val);
    }    

    void convert_yuv_to_rgb_hwc(
        uint8_t *ptr_yuv,
        int yuv_type,
        uint8_t *ptr_rgb,
        int width,
        int height,
        void *stream
    ){
        cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
        dim3 block_dim{BLOCK_SIZE, BLOCK_SIZE};
        dim3 grid_dim{
            (static_cast<uint32_t>(width) + BLOCK_SIZE - 1) / BLOCK_SIZE, 
            (static_cast<uint32_t>(height) + BLOCK_SIZE - 1) / BLOCK_SIZE
        };
        AVPixelFormat pix_fmt = (AVPixelFormat)yuv_type;
        switch (pix_fmt)
        {
        case AV_PIX_FMT_YUV420P:  // video stream
            CHECK_CUDA_KERNEL(
                convert_yuv420p_to_rgb_hwc_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_rgb, ptr_yuv, width, height
                )
            );
            break;
        case AV_PIX_FMT_YUVJ420P:  // video file
            CHECK_CUDA_KERNEL(
                convert_yuvj420p_to_rgb_hwc_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_rgb, ptr_yuv, width, height
                )
            );
            break;
        case AV_PIX_FMT_YUV422P:  // video stream
            CHECK_CUDA_KERNEL(
                convert_yuv422p_to_rgb_hwc_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_rgb, ptr_yuv, width, height
                )
            );
            break;
        case AV_PIX_FMT_YUVJ422P:  // labtop video device
            CHECK_CUDA_KERNEL(
                convert_yuvj422p_to_rgb_hwc_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_rgb, ptr_yuv, width, height
                )
            );
            break;     
        case AV_PIX_FMT_NV12:
            CHECK_CUDA_KERNEL(
                convert_nv12_to_rgb_hwc_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_rgb, ptr_yuv, width, height
                )
            );
            break;         
        default:
            INFO_FATAL("Unsupported pixel format %d", pix_fmt);
            break;
        }
    }

    __global__ void convert_bgr_hwc_rgb_hwc_kernel(
        uint8_t *ptr_bgr,
        uint8_t *ptr_rgb,
        int width,
        int height    
    ){
        // HWC stride: [WC, C, 1]
        uint32_t x_idx = blockDim.x * blockIdx.x + threadIdx.x;
        uint32_t y_idx = blockDim.y * blockIdx.y + threadIdx.y;

        uint8_t *ptr_src_b = y_idx * width * 3 + x_idx * 3 + ptr_bgr;
        uint8_t *ptr_src_g = ptr_src_b + 1;
        uint8_t *ptr_src_r = ptr_src_g + 1;

        uint8_t *ptr_dst_r = y_idx * width * 3 + x_idx * 3 + ptr_rgb;
        uint8_t *ptr_dst_g = ptr_dst_r + 1;
        uint8_t *ptr_dst_b = ptr_dst_g + 1;

        *ptr_dst_r = *ptr_src_r;
        *ptr_dst_g = *ptr_src_g;
        *ptr_dst_b = *ptr_src_b;
    }
    
    __host__ void convert_bgr_to_rgb(
        uint8_t *ptr_bgr,
        int bgr_channel_type,
        uint8_t *ptr_rgb,
        int rgb_channel_type,
        int width,
        int height,
        void *stream
    ){
        dim3 block_dim = {BLOCK_SIZE, BLOCK_SIZE};
        dim3 grid_dim = {(width - 1 + block_dim.x) / block_dim.x, (height - 1 + block_dim.y) / block_dim.y};
        cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
        switch (bgr_channel_type | rgb_channel_type)
        {
        case 1:  // HWC 2 HWC
            CHECK_CUDA_KERNEL(
                convert_bgr_hwc_rgb_hwc_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_bgr, ptr_rgb, width, height
                );
            )
            break;
        
        default:
            INFO_WARNING("Coming soon for the rest type conversion :)");
            break;
        }
    }
};