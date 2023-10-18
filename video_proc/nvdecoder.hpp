#ifndef NVENCODER_HPP
#define NVENCODER_HPP
#include <cuda.h>
#include <nvcuvid.h>
#include <utils/utils.h>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <tuple>
#include <vector>
#include "frame_cache.hpp"
struct Rect {
    int l, t, r, b;
};

struct Dim {
    int w, h;
};

class NVDecoder
{
public:
    virtual int get_width() const = 0;
    virtual int get_height() const = 0;
    virtual int decode(const uint8_t *data, int size, int flags = 0, int64_t timestamp = 0) = 0;
    virtual Frame get_frame() = 0;
    virtual FrameCache &get_frame_cache() = 0;
};

std::shared_ptr<NVDecoder> create_hardware_decoder(
    cudaVideoCodec codec, 
    int device_id = 0,
    bool use_device_frame = false, 
    bool low_latency = false, 
    bool device_frame_pitched = false,
    const Rect *crop_area = nullptr,
    const Dim *resize_dim = nullptr,
    int max_width = 0,
    int max_height = 0,
    uint32_t clock_rate = 1000,
    int cache_size = 10
);

#endif  // NVENCODER_HPP