#ifndef CVT_COLOR_CUH
#define CVT_COLOR_CUH
#include <mix_memory/mix_mat.hpp>

namespace tinycv
{
    // y u v range [0, 255]
    __device__ __inline__ void yuvj_map_to_rgb(uint8_t *r, uint8_t *g, uint8_t *b, float y, float u , float v);

    // y range [16, 235] u b range [16, 239]
    __device__ __inline__ void yuv_map_to_rgb(uint8_t *r, uint8_t *g, uint8_t *b, float y, float u , float v);

    /**
     * @brief A series of 420p convert function, include yuv420p, yuvj420p, support hwc result.
     */
    __global__ void convert_yuv420p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    );

    __global__ void convert_yuvj420p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    );

    /**
     * @brief A series of 422p convert function, include yuv422p, yuvj422p, support hwc result.
     */
    __global__ void convert_yuvj422p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    );

    __global__ void convert_yuv422p_to_rgb_hwc_kernel(
        uint8_t *dst_data, uint8_t *src_data, int width, int height
    );

    __host__ void convert_yuv_to_rgb_hwc(
        uint8_t *ptr_yuv,
        int yuv_type,
        uint8_t *ptr_rgb,
        int width,
        int height,
        void *stream = nullptr
    );

    /**
     * @brief A series of bgr to rgb convert function.
     * @todo Support HWC format only.
     */
    enum ChannelType
    {
        CHANNEL_TYPE_HWC = 1,
        CHANNEL_TYPE_CHW = 2
    };

    __global__ void convert_bgr_hwc_rgb_hwc_kernel(
        uint8_t *ptr_bgr,
        uint8_t *ptr_rgb,
        int width,
        int height    
    );

    __host__ void convert_bgr_to_rgb(
        uint8_t *ptr_bgr,
        int bgr_channel_type,
        uint8_t *ptr_rgb,
        int rgb_channel_type,
        int width,
        int height,
        void *stream = nullptr
    );
}


#endif  // CVT_COLOR_CUH
