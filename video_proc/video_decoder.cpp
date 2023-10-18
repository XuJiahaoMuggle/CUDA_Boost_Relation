#include "video_decoder.hpp"
#include "frame_cache.hpp"
#include <cstdlib>
#include <thread>
#include <string>
#include <utils/check.h>
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#ifdef __cplusplus
}
#endif

class SoftWareDecoderImpl: public SoftWareDecoder
{
public:
    virtual size_t get_decoded_frames() override {return n_decoded_frames_;}

    virtual int get_height() override { return height_;}
    
    virtual int get_width() override {return width_;}
    
    virtual int get_pixel_format() override {return pix_fmt_;}
    
    virtual FrameCache &get_frame_cache() {return frame_cache_;}

    ~SoftWareDecoderImpl() { close();}

    bool open(int codec_id, const AVCodecParameters *par, int n_threads = 4, int cache_size = 10)
    {
        codec_id_ = static_cast<AVCodecID>(codec_id);
        codec_ = const_cast<AVCodec *>(avcodec_find_decoder(codec_id_));
        if (!codec_)
            return false;
        codec_ctx_ = avcodec_alloc_context3(codec_);
        if (!codec_ctx_)
            return false;
        codec_ctx_->thread_count = n_threads > std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : n_threads;
        CHECK_FFMPEG(avcodec_parameters_to_context(codec_ctx_, par));
        CHECK_FFMPEG(avcodec_open2(codec_ctx_, nullptr, nullptr));

        pkt_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        frame_cache_ = FrameCache(cache_size);
        pix_fmt_ = static_cast<AVPixelFormat>(par->format);
        height_ = par->height;
        width_ = par->width;
        return true;
    }

    virtual int decode(uint8_t *in_buf, size_t bytes) override
    {
        if (!codec_ctx_)
            return -1;
        if (!codec_parser_ctx_ )
        {
            codec_parser_ctx_ = av_parser_init(codec_id_);
            if (!codec_parser_ctx_)
                return -1;
        }
        uint8_t *p_buff = in_buf;
        int n_decode_frames = 0; 
        while (bytes > 0)
        {
            int read_bytes = av_parser_parse2(
                codec_parser_ctx_, codec_ctx_, 
                &pkt_->data, &pkt_->size, 
                p_buff, bytes, 
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE
            );
            bytes -= read_bytes;
            p_buff += read_bytes;
            if (pkt_->size)  // enable to consist a frame from h264 raw data.
            {
                CHECK_FFMPEG(avcodec_send_packet(codec_ctx_, pkt_));  // send packet to decode.
                while (!avcodec_receive_frame(codec_ctx_, frame_))  // receive decoded frame.
                {
                    // TODO: Handle decoded frame.
                    ++n_decoded_frames_;
                    ++n_decode_frames;
                }
                av_packet_unref(pkt_);
            }
        }
        return n_decode_frames;
    }

    virtual int decode(AVPacket *in_pkt) override
    {
        if (!codec_ctx_)
            return false;
        int n_decoded_frames = 0;
        int re = avcodec_send_packet(codec_ctx_, in_pkt);
        if (re == AVERROR_EOF)
            return n_decoded_frames;
        while (!avcodec_receive_frame(codec_ctx_, frame_))  // memory padding happened here.
        {
            ++n_decoded_frames_;
            ++n_decoded_frames;

            size_t bytes = 0;
            int u_height = 0;
            int v_height = 0;
            int u_width = 0;
            int v_width = 0;

            switch (frame_->format)
            {
            case AV_PIX_FMT_YUVJ420P: // data range: YUV [0, 255]
            case AV_PIX_FMT_YUV420P:  // data range: Y [16, 235], UV [16, 239]
                bytes = height_ * width_ * 3 / 2;
                u_height = height_ / 2;
                v_height = height_ / 2;
                u_width = width_ / 2;
                v_width = width_ / 2;
                break;
            case AV_PIX_FMT_YUV422P:  // data range: Y [16, 235], UV [16, 239]
            case AV_PIX_FMT_YUVJ422P:  // data range: YUV [0, 255]
                bytes = height_ * width_ * 2;
                u_height = height_;
                v_height = height_;
                u_width = width_ / 2;
                v_width = width_ / 2;
                break;
            default:
                INFO_FATAL("Unknown decode type.");
                break;
            }

            // It's soft-decode, so we do not need to worry about the memory copy, and memcpy is much faster than cudaMemcpy-Host-Host.
            void *yuv_data = malloc(bytes);
            uint8_t *p_data = (uint8_t *)yuv_data;  // dst
            uint8_t *y_data = frame_->data[0];  // copy y plane data
            for (int i = 0; i < height_; ++i, p_data += width_, y_data += frame_->linesize[0])  
                memcpy(p_data, y_data, width_);
            uint8_t *u_data = frame_->data[1];  // copy u plane data
            for (int i = 0; i < u_height; ++i, p_data += u_width, u_data += frame_->linesize[1])  
                memcpy(p_data, u_data, u_width);
            uint8_t *v_data = frame_->data[2];  // copy v plane data
            for (int i = 0; i < v_height; ++i, p_data += v_width, v_data += frame_->linesize[2]) 
                memcpy(p_data, v_data, v_width);

            Frame frame(yuv_data, bytes, MemoryType::MEM_UNREGISTED, width_, height_, pix_fmt_, frame_->pts, -1);
            frame_cache_.push(frame);
        }
        return n_decoded_frames;
    }

    void close()
    {
        if (codec_ctx_)
            avcodec_free_context(&codec_ctx_);
        if (codec_parser_ctx_)
            av_parser_close(codec_parser_ctx_);
        if (pkt_)
            av_packet_free(&pkt_);
        if (frame_)
            av_frame_free(&frame_);
        frame_cache_.clear();
    }

private:
    AVCodecID codec_id_ = AV_CODEC_ID_NONE;
    AVCodec *codec_ = nullptr;
    AVCodecContext *codec_ctx_ = nullptr;
    AVCodecParserContext *codec_parser_ctx_ = nullptr;

    AVPixelFormat pix_fmt_ = AV_PIX_FMT_NONE;
    int height_ = 0;
    int width_ = 0;

    AVFrame *frame_ = nullptr;
    AVPacket *pkt_ = nullptr;
    size_t n_decoded_frames_ = 0;

    FrameCache frame_cache_;
};

std::shared_ptr<SoftWareDecoder> create_software_decoder(int codec_id, const AVCodecParameters *par, int n_threads, int cache_size)
{
    std::shared_ptr<SoftWareDecoderImpl> instance = std::make_shared<SoftWareDecoderImpl>();
    if (!instance->open(codec_id, par, n_threads, cache_size))
        instance.reset();
    return instance;
}