#ifndef VIDEO_PROC_H
#define VIDEO_PROC_H

#include "demuxer.hpp"
#include "video_decoder.hpp"
#include "frame_cache.hpp"
#include "nvdecoder.hpp"
#include <mix_memory/mix_mat.hpp>
#include <memory>
#include <string>

struct AVPacket;
namespace tinycv
{
    class VideoCap
    {
    public:
        VideoCap() = default;
        ~VideoCap();
        VideoCap(const std::string &url, const std::string &in_fmt_name, size_t time_scale = 1000);
        bool open(const std::string &url, const std::string &in_fmt_name, size_t time_scale = 1000);
        
        void operator>>(Frame &frame);
        Frame read_frame();
        MixMat read_mix_mat(void *stream = nullptr);

        int get_height();
        int get_width();
        int get_n_frames();

    protected:
        std::shared_ptr<Demuxer> demuxer_ = nullptr;
        std::shared_ptr<BaseMemory> read_buffer_ = nullptr;
        AVPacket *pkt_;
        int64_t pts_;

    private:
        std::shared_ptr<SoftWareDecoder> soft_decoder_ = nullptr;
    };

    class NVVideoCap
    {
    public:
        NVVideoCap() = default;
        ~NVVideoCap();
        NVVideoCap(
            const std::string &url, 
            const std::string &in_fmt_name, 
            int device_id,
            bool use_device_frame, 
            size_t time_scale = 1000
        );

        bool open(
            const std::string &url, 
            const std::string &in_fmt_name, 
            int device_id,
            bool use_device_frame, 
            size_t time_scale = 1000
        );
        
        void operator>>(Frame &frame);
        Frame read_frame();
        MixMat read_mix_mat(void *stream = nullptr);

        int get_height();
        int get_width();
        int get_n_frames();

    protected:
        std::shared_ptr<Demuxer> demuxer_ = nullptr;
        std::shared_ptr<BaseMemory> read_buffer_ = nullptr;
        AVPacket *pkt_;
        int64_t pts_;
    
    private:
        std::shared_ptr<NVDecoder> decoder_ = nullptr;
    };
}
#endif  // VIDEO_PROC_H