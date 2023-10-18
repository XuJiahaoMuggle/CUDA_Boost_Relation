#include "video_proc.h"
#include <thread>
#ifdef __cplusplus
extern "C"
{
#endif    
#include <libavcodec/avcodec.h>
#ifdef __cplusplus
}
#endif
namespace tinycv
{
    VideoCap::VideoCap(const std::string &url, const std::string &in_fmt_name, size_t time_scale)
    {
        open(url, in_fmt_name, time_scale);
    }

    VideoCap::~VideoCap()
    {
        if (pkt_)
            av_packet_free(&pkt_);
    }

    bool VideoCap::open(const std::string &url, const std::string &in_fmt_name, size_t time_scale)
    {
        demuxer_ = create_demuxer(url, in_fmt_name, time_scale);
        if (demuxer_ == nullptr)
            return false;
        soft_decoder_ = create_software_decoder(
            demuxer_->get_video_codec_id(), 
            demuxer_->get_video_codec_par(),
            std::thread::hardware_concurrency(),
            10
        );
        if (soft_decoder_ == nullptr)
            return false;
        pkt_ = av_packet_alloc();
        return true;
    }

    void VideoCap::operator>>(Frame &frame)
    {
        FrameCache &frame_cache = soft_decoder_->get_frame_cache();
        while (frame_cache.empty())
        {
            bool demux_flag = demuxer_->demux(pkt_, &pts_);  // get squeezed frame (packet)
            int n_decoded_frames = soft_decoder_->decode(pkt_);  // decode frame, which may need several squeezed frame.
            if (!demux_flag && n_decoded_frames == 0)  // no more squeezed frame && the frame cache is empty.
                break;
            av_packet_unref(pkt_);
        }
        if (frame_cache.empty())
            frame = {};
        else
        {
            frame = frame_cache.front();
            frame_cache.pop();
        }
    }

    Frame VideoCap::read_frame()
    {
        Frame frame;
        this->operator>>(frame);
        return frame;
    }

    MixMat VideoCap::read_mix_mat(void *stream)
    {
        Frame frame;
        this->operator>>(frame);
        return make_mix_mat_from_frame(frame, read_buffer_, stream);
    }

    int VideoCap::get_height()
    {
        if (demuxer_)
            return demuxer_->get_height();
        return -1;
    }

    int VideoCap::get_width()
    {
        if (demuxer_)
            return demuxer_->get_width();
        return -1;
    }

    int VideoCap::get_n_frames()
    {
        if (demuxer_)
            return demuxer_->get_total_frames();
        return -1;
    }

    NVVideoCap::~NVVideoCap()
    {
        if (pkt_)
            av_packet_free(&pkt_);
    }

    NVVideoCap::NVVideoCap(        
        const std::string &url, 
        const std::string &in_fmt_name, 
        int device_id,
        bool use_device_frame, 
        size_t time_scale
    ) { open(url, in_fmt_name, device_id, use_device_frame, time_scale); }

    bool NVVideoCap::open(
        const std::string &url, 
        const std::string &in_fmt_name, 
        int device_id,
        bool use_device_frame, 
        size_t time_scale
    ) {
        demuxer_ = create_demuxer(url, in_fmt_name, time_scale);
        if (demuxer_ == nullptr)
            return false;
        decoder_ = create_hardware_decoder(avcodec_to_nvcodec_id(demuxer_->get_video_codec_id()), device_id, use_device_frame);
        if (decoder_ == nullptr)
            return false;
        pkt_ = av_packet_alloc();
        return true;
    }
        
    void NVVideoCap::operator>>(Frame &frame)
    {
        FrameCache &frame_cache = decoder_->get_frame_cache();
        while (frame_cache.empty())
        {
            size_t pkt_data_size = 0;
            bool demux_flag = demuxer_->demux(pkt_, &pts_);  // get squeezed frame (packet)
            int n_decoded_frames = decoder_->decode(pkt_->data, pkt_->size);  // decode frame, which may need several squeezed frame.
            if (!demux_flag && n_decoded_frames == 0)  // no more squeezed frame && the frame cache is empty.
                break;
            av_packet_unref(pkt_);
        }
        if (frame_cache.empty())
            frame = {};
        else
        {
            frame = frame_cache.front();
            frame_cache.pop();
        }
    }

    Frame NVVideoCap::read_frame()
    {
        Frame frame;
        this->operator>>(frame);
        return frame;
    }

    MixMat NVVideoCap::read_mix_mat(void *stream) 
    { 
        Frame frame;
        this->operator>>(frame);
        return make_mix_mat_from_frame(frame, read_buffer_, stream);
    }

    int NVVideoCap::get_height() { return demuxer_->get_height(); }
    int NVVideoCap::get_width() { return demuxer_->get_width(); }
    int NVVideoCap::get_n_frames() { return demuxer_->get_total_frames(); }

}