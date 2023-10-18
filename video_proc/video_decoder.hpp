#ifndef VIDEO_DECODER_HPP
#define VIDEO_DECODER_HPP
#include <memory>
#include "frame_cache.hpp"

struct AVPacket;
struct AVCodecParameters;

// RAII
class SoftWareDecoder
{
public:
    virtual size_t get_decoded_frames() = 0;
    virtual int get_height() = 0;
    virtual int get_width() = 0;
    virtual int get_pixel_format() = 0;
    virtual int decode(uint8_t *in_buf, size_t bytes) = 0;
    virtual int decode(AVPacket *in_pkt) = 0;
    virtual FrameCache &get_frame_cache() = 0;
};

std::shared_ptr<SoftWareDecoder> create_software_decoder(
    int codec_id, const AVCodecParameters *par, 
    int n_threads = 4, int cache_size = 10
);

#endif  // VIDEO_DECODER_HPP