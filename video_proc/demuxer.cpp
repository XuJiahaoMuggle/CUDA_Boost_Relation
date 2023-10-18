#include "demuxer.hpp"

#include <utils/check.h>
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#ifdef __cplusplus
}
#endif

#include <string>
#include <iostream>

cudaVideoCodec avcodec_to_nvcodec_id(int id) 
{
    switch ((AVCodecID)id) {
    case AV_CODEC_ID_MPEG1VIDEO : return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO : return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4      : return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_VC1        : return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264       : return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC       : return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8        : return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9        : return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG      : return cudaVideoCodec_JPEG;
    default                     : return cudaVideoCodec_NumCodecs;
    }
}

class DemuxerImpl: public Demuxer
{
public:
    virtual int get_video_codec_id() override
    {
        return video_codec_id_;
    }

    virtual int get_pixel_format() override
    {
        return pix_fmt_;
    }

    virtual int get_height() override
    {
        return height_;
    }

    virtual int get_width() override
    {
        return width_;
    }

    virtual int get_bit_depth() override
    {
        return bit_depth_;
    }

    virtual size_t get_total_frames() override
    {
        return n_total_frames_;
    }

    virtual int get_fps() override
    {
        return fps_;
    }

    virtual void get_extra_data(uint8_t **pp_data, int *p_bytes) override
    {
        *pp_data = format_ctx_->streams[video_stream_idx_]->codecpar->extradata;
        *p_bytes = format_ctx_->streams[video_stream_idx_]->codecpar->extradata_size;
    }

    virtual const AVCodecParameters *get_video_codec_par() override
    {
        if (!format_ctx_)
            return nullptr;
        return format_ctx_->streams[video_stream_idx_]->codecpar;
    }

    virtual bool demux(uint8_t **nalu_data, size_t *p_size, int64_t *pts, bool *is_key_frame) override
    {
        *nalu_data = nullptr;
        *p_size = 0;
        if (!format_ctx_)
            return false;
        if (pkt_->data)
            av_packet_unref(pkt_);
        int re = 0;
        while ((re = av_read_frame(format_ctx_, pkt_)) >= 0 && video_stream_idx_ != pkt_->stream_index)
            av_packet_unref(pkt_);
        if (re < 0)
            return false;
        if (is_key_frame)
            *is_key_frame = pkt_->flags & AV_PKT_FLAG_KEY;
        int64_t local_pts = 0;
        if (is_mp4_h264_ || is_mp4_hevc_)
        {
            if (pkt_filtered_->data)
                av_packet_unref(pkt_filtered_);
            CHECK_FFMPEG(av_bsf_send_packet(bsf_ctx_, pkt_));
            CHECK_FFMPEG(av_bsf_receive_packet(bsf_ctx_, pkt_filtered_));
            *nalu_data = pkt_filtered_->data;
            *p_size = pkt_filtered_->size;
            local_pts = (int64_t)(pkt_filtered_->pts * time_scale_ * time_base_);
        }
        else if (is_mp4_mpeg4_ && (n_counted_frames_ == 0))
        {
            int n_extra_data_bytes = format_ctx_->streams[video_stream_idx_]->codecpar->extradata_size;
            if (n_extra_data_bytes > 0)
            {
                // exrta_data: [start code] 0x00 0x00 0x01 [data1]
                // pkt_->data: [start code] 0x00 0x00 0x01 [data2] 
                // head_data_:[start code] 0x00 0x00 0x01 [data1] [data2]
                head_data_ = (uint8_t *)av_malloc(n_extra_data_bytes + pkt_->size - 3 * sizeof(uint8_t));
                memcpy(head_data_, format_ctx_->streams[video_stream_idx_]->codecpar->extradata, n_extra_data_bytes);
                memcpy(head_data_ + n_extra_data_bytes, pkt_->data + 3, pkt_->size - 3 * sizeof(uint8_t));
                *nalu_data = head_data_;
                *p_size = n_extra_data_bytes + pkt_->size - 3 * sizeof(uint8_t);
            }
            local_pts = (int64_t)(pkt_->pts * time_scale_ * time_base_);
        }
        else
        {
            *nalu_data = pkt_->data;
            *p_size = pkt_->size;
            local_pts = (int64_t)(pkt_->pts * time_scale_ * time_base_);
        }
        if (pts)
            *pts = local_pts;
       ++n_counted_frames_;
       return true; 
    }

    virtual bool demux(AVPacket *pkt, int64_t *pts, bool *is_key_frame) override
    {
        if (!format_ctx_)
            return false;
        if (pkt_->data)
            av_packet_unref(pkt_);
        int re = 0;
        while ((re = av_read_frame(format_ctx_, pkt_)) >= 0 && video_stream_idx_ != pkt_->stream_index)
            av_packet_unref(pkt_);
        if (re < 0)
            return false;
        if (is_key_frame)
            *is_key_frame = pkt_->flags & AV_PKT_FLAG_KEY;
        int64_t local_pts = 0;
        if (is_mp4_h264_ || is_mp4_hevc_)
        {
            if (pkt_filtered_->data)
                av_packet_unref(pkt_filtered_);
            CHECK_FFMPEG(av_bsf_send_packet(bsf_ctx_, pkt_));
            CHECK_FFMPEG(av_bsf_receive_packet(bsf_ctx_, pkt_filtered_));
            local_pts = (int64_t)(pkt_filtered_->pts * time_scale_ * time_base_);
            CHECK_FFMPEG(av_packet_ref(pkt, pkt_filtered_));
        }
        else
        {
            CHECK_FFMPEG(av_packet_ref(pkt, pkt_));
            local_pts = (int64_t)(pkt_->pts * time_scale_ * time_base_);
        }
        if (pts)
            *pts = local_pts;
       ++n_counted_frames_;

       return true;         
    }

    ~DemuxerImpl() { close(); }

    void close()
    {
        if (pkt_)
            av_packet_free(&pkt_);
        if (pkt_filtered_)
            av_packet_free(&pkt_filtered_);
        if (bsf_ctx_)
            av_bsf_free(&bsf_ctx_);
        if (format_ctx_)
            avformat_close_input(&format_ctx_);
        if (head_data_)
            av_free(head_data_);
        is_opened_ = false;
    }

    bool open(const std::string &file, const std::string &input_fmt_name, size_t time_scale = 1000)
    {
        file_ = file;
        time_scale_ = time_scale;
        const AVInputFormat *input_fmt = nullptr;
        AVDictionary *opts = nullptr;
        if (!input_fmt_name.empty())
        {
            avdevice_register_all();
            // av_dict_set_int(&opts, "rtbufsize", 18432000, 0);
            input_fmt = av_find_input_format(input_fmt_name.c_str());
        }
        CHECK_FFMPEG(avformat_open_input(&format_ctx_, file.c_str(), input_fmt, &opts));
        if (!format_ctx_)
            return false;
        CHECK_FFMPEG(avformat_find_stream_info(format_ctx_, nullptr));
        for (int i = 0; i < format_ctx_->nb_streams; ++i)
        {
            if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                video_stream_idx_ = i;
                video_codec_id_ = format_ctx_->streams[i]->codecpar->codec_id;
                height_ = format_ctx_->streams[i]->codecpar->height;
                width_ = format_ctx_->streams[i]->codecpar->width;
                pix_fmt_ = (AVPixelFormat)format_ctx_->streams[i]->codecpar->format;
                time_base_ = av_q2d(format_ctx_->streams[i]->time_base);
                time_scale_ = time_scale;
                AVRational fps = format_ctx_->streams[i]->avg_frame_rate;
                fps_ = fps.num == 0 || fps.den == 0 ? 0 : av_q2d(fps);
                n_total_frames_ = format_ctx_->streams[i]->nb_frames;
                break;
            }
        }
        if (video_stream_idx_ < 0)
        {
            INFO_ERROR("Can not find video stream info!");
            return false;
        }
        switch (pix_fmt_)
        {
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_YUV420P:
            bit_depth_ = 8;
            chroma_height_ = (height_ + 1) >> 1;
            break;

        case AV_PIX_FMT_YUVJ422P:
        case AV_PIX_FMT_YUV422P:
            bit_depth_ = 8;
            chroma_height_ = height_;
            break;
        default:
            break;
        }
        is_mp4_h264_ = video_codec_id_ == AV_CODEC_ID_H264 && (
            !strcmp(format_ctx_->iformat->long_name, "QuickTime / MOV")
            || !strcmp(format_ctx_->iformat->long_name, "FLV (Flash Video)")
            || !strcmp(format_ctx_->iformat->long_name, "Matroska / WebM")
        );
        is_mp4_hevc_ = video_codec_id_ == AV_CODEC_ID_HEVC && (
            !strcmp(format_ctx_->iformat->long_name, "QuickTime / MOV")
            || !strcmp(format_ctx_->iformat->long_name, "FLV (Flash Video)")
            || !strcmp(format_ctx_->iformat->long_name, "Matroska / WebM")
        );
        is_mp4_mpeg4_ = video_codec_id_ == AV_CODEC_ID_MPEG4 && (
            !strcmp(format_ctx_->iformat->long_name, "QuickTime / MOV")
            || !strcmp(format_ctx_->iformat->long_name, "FLV (Flash Video)")
            || !strcmp(format_ctx_->iformat->long_name, "Matroska / WebM")
        );
        // initialize AVPacket.
        pkt_ = av_packet_alloc();
        av_init_packet(pkt_);
        pkt_->data = nullptr;
        pkt_->size = 0;

        pkt_filtered_ = av_packet_alloc();
        av_init_packet(pkt_filtered_);
        pkt_filtered_->data = nullptr;
        pkt_filtered_->size = 0;
        
        const AVBitStreamFilter *bsf = nullptr;
        if (is_mp4_h264_)
        {
            bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if (!bsf)
            {
                INFO_ERROR("av_bsf_get_by_name() failed");
                return false;
            }
        }
        else if (is_mp4_hevc_)
        {
            bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            if (!bsf)
            {
                INFO_ERROR("av_bsf_get_by_name() failed");
                return false;
            }
        }
        if (bsf)
        {
            if (!CHECK_FFMPEG(av_bsf_alloc(bsf, &bsf_ctx_)))
                return false;
            if (!CHECK_FFMPEG(avcodec_parameters_copy(bsf_ctx_->par_in, format_ctx_->streams[video_stream_idx_]->codecpar)))
                return false;
            if (!CHECK_FFMPEG(av_bsf_init(bsf_ctx_)))
                return false;
        }
        is_opened_ = true;
        return true;
    }

private:
    std::string file_; 
    AVFormatContext *format_ctx_ = nullptr;
    AVCodecID video_codec_id_ = AV_CODEC_ID_NONE;

    size_t n_counted_frames_ = 0;
    size_t n_total_frames_ = 0;
    size_t time_scale_ = 0;
    double time_base_ = 0.0;

    int height_ = 0;
    int width_ = 0;
    int bit_depth_ = 0;
    int fps_ = 0;
    int chroma_height_ = 0;

    int video_stream_idx_ = -1;
    AVPixelFormat pix_fmt_ = AV_PIX_FMT_NONE;

    // AVCC -> AnnexB flag
    bool is_mp4_h264_ = false;
    bool is_mp4_hevc_ = false;
    bool is_mp4_mpeg4_ = false;

    AVPacket *pkt_ = nullptr;
    AVPacket *pkt_filtered_ = nullptr;
    AVBSFContext *bsf_ctx_ = nullptr;

    uint8_t *head_data_ = nullptr;
    bool is_opened_ = false;
};

std::shared_ptr<Demuxer> create_demuxer(const std::string &file, const std::string &input_fmt_name, size_t time_scale)
{
    std::shared_ptr<DemuxerImpl> instance = std::make_shared<DemuxerImpl>();
    if (!instance->open(file, input_fmt_name, time_scale))
        instance.reset();
    return instance;
}
