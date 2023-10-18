#ifndef DEMUXER_HPP
#define DEMUXER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <nvcuvid.h>

#ifdef __cplusplus
extern "C" 
{
#endif 

#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>

#ifdef __cplusplus
}
#endif

struct AVPacket;
struct AVCodecParameters;

cudaVideoCodec avcodec_to_nvcodec_id(int id);

// RAII 
class Demuxer
{
public:
    virtual int get_video_codec_id() = 0;
    virtual int get_pixel_format() = 0;
    virtual int get_height() = 0;
    virtual int get_width() = 0;
    virtual int get_bit_depth() = 0;
    virtual size_t get_total_frames() = 0;
    virtual int get_fps() = 0;
    virtual void get_extra_data(uint8_t **pp_data, int *p_bytes) = 0;
    virtual const AVCodecParameters *get_video_codec_par() = 0;
    virtual bool demux(uint8_t **data, size_t *size, int64_t *pts = nullptr, bool *is_key_frame = nullptr) = 0;
    virtual bool demux(AVPacket *pkt, int64_t *pts = nullptr, bool *is_key_frame = nullptr) = 0;
};

std::shared_ptr<Demuxer> create_demuxer(const std::string &file, const std::string &input_fmt_name = "", size_t time_scale = 1000);

#endif  // DEMUXER_HPP