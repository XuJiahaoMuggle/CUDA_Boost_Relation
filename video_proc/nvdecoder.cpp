#include "nvdecoder.hpp"
#include <utils/utils.h>
#ifdef __cplusplus
extern "C"
{
#endif 
#include <libavcodec/avcodec.h>
#ifdef __cplusplus
}
#endif 


static const char *get_videocodec_string(cudaVideoCodec codec_id)
{
    static struct 
    {
        cudaVideoCodec codec_id;
        const char *name;
    } codec_names[] = {
        { cudaVideoCodec_MPEG1,     "MPEG-1"       },
        { cudaVideoCodec_MPEG2,     "MPEG-2"       },
        { cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
        { cudaVideoCodec_VC1,       "VC-1/WMV"     },
        { cudaVideoCodec_H264,      "AVC/H.264"    },
        { cudaVideoCodec_JPEG,      "M-JPEG"       },
        { cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
        { cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
        { cudaVideoCodec_HEVC,      "H.265/HEVC"   },
        { cudaVideoCodec_VP8,       "VP8"          },
        { cudaVideoCodec_VP9,       "VP9"          },
        { cudaVideoCodec_NumCodecs, "Invalid"      },
        { cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
        { cudaVideoCodec_YV12,      "YV12 4:2:0"   },
        { cudaVideoCodec_NV12,      "NV12 4:2:0"   },
        { cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
        { cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
    };
    if (codec_id >= 0 && codec_id <= cudaVideoCodec_NumCodecs)
        return codec_names[codec_id].name;
    for (int i = cudaVideoCodec_NumCodecs + 1; i < sizeof(codec_names) / sizeof(codec_names[0]); i++) 
    {
        if (codec_id == codec_names[i].codec_id) 
        {
            return codec_names[codec_id].name;
        }
    }
    return "Unknown";
}

static const char *get_video_chroma_format_string(cudaVideoChromaFormat chroma_format) 
{
    static struct 
    {
        cudaVideoChromaFormat chroma_format;
        const char *name;
    } chroma_format_names[] = {
        { cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { cudaVideoChromaFormat_420,        "YUV 420"              },
        { cudaVideoChromaFormat_422,        "YUV 422"              },
        { cudaVideoChromaFormat_444,        "YUV 444"              },
    };

    if (chroma_format >= 0 && chroma_format < sizeof(chroma_format_names) / sizeof(chroma_format_names[0])) 
        return chroma_format_names[chroma_format].name;
    return "Unknown";
}

static float get_chroma_height_factor(cudaVideoSurfaceFormat surface_format)
{
    float factor = 0.5;
    switch (surface_format)
    {
    case cudaVideoSurfaceFormat_NV12:  
    case cudaVideoSurfaceFormat_P016:  
        factor = 0.5;
        break;
    case cudaVideoSurfaceFormat_YUV444:
    case cudaVideoSurfaceFormat_YUV444_16Bit:
        factor = 1.0;
        break;
    }
    return factor;
}

static int get_chromal_plane_cnt(cudaVideoSurfaceFormat surface_format)
{
    int n_planes = 1;
    switch (surface_format)
    {
    case cudaVideoSurfaceFormat_NV12:  
    case cudaVideoSurfaceFormat_P016:  
        n_planes = 1;
        break;
    case cudaVideoSurfaceFormat_YUV444:  
    case cudaVideoSurfaceFormat_YUV444_16Bit:
        n_planes = 2;
        break;
    }
    return n_planes;
}

static int nvformat_to_avpixfomat(cudaVideoSurfaceFormat surface_format)
{
    switch (surface_format)
    {
    case cudaVideoSurfaceFormat_NV12:  
        return AVPixelFormat::AV_PIX_FMT_NV12;
    case cudaVideoSurfaceFormat_P016:  
        return AVPixelFormat::AV_PIX_FMT_NV16;
    case cudaVideoSurfaceFormat_YUV444:  
        return AVPixelFormat::AV_PIX_FMT_YUV444P;
    case cudaVideoSurfaceFormat_YUV444_16Bit:
        return AVPixelFormat::AV_PIX_FMT_YUV444P16LE;
    default:
        return AVPixelFormat::AV_PIX_FMT_NONE;
    }
}

class NVDecoderImpl: public NVDecoder
{
public: 
    bool create(
        cudaVideoCodec codec,
        int device_id,
        bool use_device_frame, 
        bool low_latency = false, 
        bool device_frame_pitched = false,
        const Rect *crop_area = nullptr,
        const Dim *resize_dim = nullptr,
        int max_width = 0,
        int max_height = 0,
        uint32_t clock_rate = 1000,
        int cache_size = 10
    );

    ~NVDecoderImpl();

    /// @brief Return the cuda context used for decoder. 
    CUcontext get_cuda_ctx() const { return cu_ctx_; }

    /// @brief Return the width of image. 
    int get_width() const override { return width_; }

    /// @brief Return the luma height of frame(Y Plane)
    int get_height() const override { return luma_height_; }

    /// @brief Return the chroma height of frame(UV Plane) 
    int get_chroma_height() const { return chroma_height_; }

    /// @brief Return the number of chroma planes. 
    int get_nb_chroma_planes() const { return n_chroma_planes_; }

    /// @brief Return the number of byets of one frame.
    int get_frame_size() const { return width_ * (luma_height_ + chroma_height_ * n_chroma_planes_) * bytes_per_pixel_; }

    /// @brief Return the number of byets of one pitched frame.
    int get_device_frame_pitch() const { return device_frame_pitched_ ? static_cast<int>(n_device_frame_pitched_) : width_ * bytes_per_pixel_; }

    /// @brief Return the bit depth.  
    int get_bit_depth() const { return bit_depth_minus8_ + 8; }

    /// @brief Return the number of bytes per pixel. 
    int get_bytes_per_pixel() const { return bytes_per_pixel_; }

    /// @brief Return the YUV chroma format.
    cudaVideoSurfaceFormat get_output_format() const { return video_surface_format_; }

    /// @brief Return the video format. 
    CUVIDEOFORMAT get_video_format_info() const { return video_format_; }

    /// @brief Return the codec string from codec id.
    /// @param codec 
    const char *get_codec_string(cudaVideoCodec codec_id);

    /// @brief Return the information about video stream. 
    std::string get_video_info() const { return video_info_.str(); }

    /// @brief Decode a frame and returns the number of frames that are available for display.
    /// @param data (const uint8_t *): the frame to be decoded.
    /// @param size (int): size of data in bytes.
    /// @param flags (int): CUvideopacketflags for setting decode options
    /// @param timestamp (int64_t): pts.
    int decode(const uint8_t *data, int size, int flags = 0, int64_t timestamp = 0);

    /// @brief Return a decoded frame and timestamp.
    Frame get_frame() override;

    FrameCache &get_frame_cache() override {return frame_cache_;};
    /// @brief Reconfig the params. 
    int set_reconfig_params(const Rect *crop_area, const Dim *resize_dim);

private:
    /// @brief Will be called when a sequence is ready to be decoded or when there is format change.
    int handle_video_sequence(CUVIDEOFORMAT *video_format);
    static int CUDAAPI HandleVideoSequenceProc(void *user_data, CUVIDEOFORMAT *video_format) { return static_cast<NVDecoderImpl *>(user_data)->handle_video_sequence(video_format); }

    /// @brief Will be called when a picture is ready be decoded.
    int handle_picture_decode(CUVIDPICPARAMS *pic_params);
    static int CUDAAPI HandlePictureDecodeProc(void *user_data, CUVIDPICPARAMS *pic_params) { return static_cast<NVDecoderImpl *>(user_data)->handle_picture_decode(pic_params); }

    /// @brief Will be called when a picture is decoded and ready to display. 
    int handle_picture_display(CUVIDPARSERDISPINFO *display_info);
    static int CUDAAPI HandlePictureDisplayProc(void *user_data, CUVIDPARSERDISPINFO *display_info) {return static_cast<NVDecoderImpl *>(user_data)->handle_picture_display(display_info); }

    int reconfig_decoder(CUVIDEOFORMAT *video_format);

private:
    CUcontext cu_ctx_ = nullptr;
    int device_id_ = -1;
    CUvideoctxlock video_ctx_lock_;
    CUvideoparser video_parser_ = nullptr;
    CUvideodecoder video_decoder_ = nullptr;
    bool use_device_frame_ = false;
    uint32_t width_ = 0, luma_height_ = 0, chroma_height_ = 0;

    uint32_t n_chroma_planes_ = 0;
    int surface_height_ = 0, surface_width_ = 0;
    cudaVideoCodec video_codec_ = cudaVideoCodec_NumCodecs;
    cudaVideoChromaFormat video_chroma_format_;
    cudaVideoSurfaceFormat video_surface_format_;

    int bit_depth_minus8_ = 0;
    int bytes_per_pixel_ = 1;
    CUVIDEOFORMAT video_format_ = {};
    Rect display_rect_ = {};

    int n_decoded_frames_ = 0;
    // int n_decoded_frames_returned_ = 0;

    int n_frames_alloc_ = 0;
    bool done_ = false;

    // std::mutex frame_mtx_;

    CUstream cuvid_stream_ = 0;
    bool device_frame_pitched_ = false;
    size_t n_device_frame_pitched_ = 0;
    Rect crop_area_ = {};
    Dim resize_dim_ = {};

    std::ostringstream video_info_;
    uint32_t max_width_ = 0, max_height_ = 0;
    
    FrameCache frame_cache_;

    bool reconfig_external_ = false;
    bool reconfig_external_pp_change_ = false;
};

const char *NVDecoderImpl::get_codec_string(cudaVideoCodec codec_id) { return get_videocodec_string(codec_id); }

bool NVDecoderImpl::create(
    cudaVideoCodec codec,
    int device_id,
    bool use_device_frame, 
    bool low_latency, 
    bool device_frame_pitched,
    const Rect *crop_area,
    const Dim *resize_dim,
    int max_width,
    int max_height,
    uint32_t clock_rate,
    int cache_size
) {
    if (cache_size == -1)
        cache_size = 10;
    if (cache_size < 0)
        return false;
    frame_cache_ = FrameCache(cache_size);
    cuda_tools::AutoExchangeDevice ex(device_id);
    if (!CHECK_CUDA_DRIVER(cuCtxGetCurrent(&cu_ctx_)))
    {
        INFO_ERROR("Invalid device id.");
        return false;
    }
    device_id_ = device_id;
    use_device_frame_ = use_device_frame;
    video_codec_ = codec;
    device_frame_pitched_ = device_frame_pitched;
    max_width_ = max_width;
    max_height_ = max_height;
    if (crop_area)
        crop_area_ = *crop_area;
    if (resize_dim)
        resize_dim_ = *resize_dim;
    // For cuvid, cause it requires multi threads to decode.
    if (!CHECK_CUDA_DRIVER(cuvidCtxLockCreate(&video_ctx_lock_, cu_ctx_)))
    {
        INFO_ERROR("Cuvid lock cuda context failed.");
        return false;   
    }
    // fill parser param, and create the video parser
    CUVIDPARSERPARAMS video_parser_params = {};
    video_parser_params.CodecType = codec; 
    // the number of surfaces in parser's decode picture buffer, may not be know now, set a dummy number. 
    video_parser_params.ulMaxNumDecodeSurfaces = 1;  
    video_parser_params.ulClockRate = clock_rate;
    // max display callback delay
    video_parser_params.ulMaxDisplayDelay = low_latency ? 0 : 1;  
    video_parser_params.pUserData = this;
    // binding callback function
    // triggers when initial sequence header or when encouters a video format change.
    video_parser_params.pfnSequenceCallback = HandleVideoSequenceProc;  
    // triggers when bitstream data for one frame is ready, which means this function is used for decoding.
    video_parser_params.pfnDecodePicture = HandlePictureDecodeProc;
    // triggers when a frame is ready for render, which means this function is used for fetch the decoded result.
    video_parser_params.pfnDisplayPicture = HandlePictureDisplayProc;
    if (!CHECK_CUDA_DRIVER(cuvidCreateVideoParser(&video_parser_, &video_parser_params)))
    {
        INFO_ERROR("Cuvid create video parser failed.");
        return false;   
    }
    return true;
}

NVDecoderImpl::~NVDecoderImpl() 
{
    if (video_parser_)
        cuvidDestroyVideoParser(video_parser_);
    if (video_decoder_)
        cuvidDestroyDecoder(video_decoder_);
    {
        // std::lock_guard<std::mutex> lock(frame_mtx_);  // thread safe opeartion.
        frame_cache_.clear();
    }
    cuvidCtxLockDestroy(video_ctx_lock_);

}

/// @brief Decode one frame.
/// @param pkt_data (const uint8_t *): the squeezed packet data.
/// @param size (int): the number of bytes of packte data.
/// @param flags (int): 0 default.
/// @param pts (int64_t): the presentation time stamp.
/// @return (int): the number of decoded frames.
int NVDecoderImpl::decode(const uint8_t *pkt_data, int size, int flags, int64_t pts)
{
    // Each time call decode reset n_decoded_frames_ and n_decoded_frames_returned_
    n_decoded_frames_ = 0;  
    // n_decoded_frames_returned_ = 0;
    CUVIDSOURCEDATAPACKET packet = {0};
    packet.payload = pkt_data;
    packet.payload_size = size;
    // set invalid timestamp flag.
    packet.flags = flags | CUVID_PKT_TIMESTAMP;  
    // set end of stream flag.
    if (!pkt_data || size == 0)
        packet.flags |= CUVID_PKT_ENDOFSTREAM;  
    // parser packet data and will call back pfnDecodePicture, pfnSequenceCallback, pfnDisplayPicture.
    cuda_tools::AutoExchangeDevice ex(device_id_);
    cuvidParseVideoData(video_parser_, &packet);  
    cuvid_stream_ = 0;
    return n_decoded_frames_;
}

Frame NVDecoderImpl::get_frame()
{
    if (n_decoded_frames_ > 0)
    {
        // std::lock_guard<std::mutex> lock(frame_mtx_);
        n_decoded_frames_--;
        Frame frame = frame_cache_.front();
        frame_cache_.pop();
        return frame;
    }
    return {};
}

/// @brief Calls back function to initial header or when encouter a video format change.
/// @param video_format (CUVIDEOFORMAT *): the changed video format.
/// @return (int): 0 on failed, >= 1 on successed.
int NVDecoderImpl::handle_video_sequence(CUVIDEOFORMAT *video_format)
{
    // record video information.
    video_info_.str("");
    video_info_.clear();
    video_info_ << "Video Input Information" << std::endl
        << "\tCodec        : " << get_videocodec_string(video_format->codec) << std::endl
        << "\tFrame rate   : " << video_format->frame_rate.numerator << "/" << video_format->frame_rate.denominator 
            << " = " << 1.0 * video_format->frame_rate.numerator / video_format->frame_rate.denominator << " fps" << std::endl
        << "\tSequence     : " << (video_format->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
        << "\tCoded size   : [" << video_format->coded_width << ", " << video_format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << video_format->display_area.left << ", " << video_format->display_area.top << ", "
            << video_format->display_area.right << ", " << video_format->display_area.bottom << "]" << std::endl
        << "\tChroma       : " << get_video_chroma_format_string(video_format->chroma_format) << std::endl
        << "\tBit depth    : " << video_format->bit_depth_luma_minus8 + 8;
    video_info_ << std::endl;
    // Obtained from first sequence callback from nvidia parser.
    int n_decoded_surfaces = video_format->min_num_decode_surfaces;  
    // To fill cuvidGetDecoderCaps
    CUVIDDECODECAPS decode_caps;
    memset(&decode_caps, 0, sizeof(decode_caps));
    decode_caps.eCodecType = video_format->codec;
    decode_caps.eChromaFormat = video_format->chroma_format;
    decode_caps.nBitDepthMinus8 = video_format->bit_depth_luma_minus8;
    // query the capabilities of underlying hardware video decoder, this requires a cuda context.
    cuvidGetDecoderCaps(&decode_caps);  
    // check if support the codec type.
    if (!decode_caps.bIsSupported)
    {
        INFO_ERROR("Unsupported codec!");
        return n_decoded_surfaces;
    }
    // Too large too decode.
    if ((video_format->coded_width > decode_caps.nMaxWidth) || (video_format->coded_height > decode_caps.nMaxHeight))
    {
        std::ostringstream error_ss;
        error_ss << std::endl
                    << "Resolution          : " << video_format->coded_width << "x" << video_format->coded_height << std::endl
                    << "Max Supported (wxh) : " << decode_caps.nMaxWidth << "x" << decode_caps.nMaxHeight << std::endl
                    << "Resolution not supported on this GPU";
        INFO_ERROR(error_ss.str().c_str());
        return n_decoded_surfaces;
    }
    // Confused why requires coded_width * coded_height / 256 <= nMaxMBCount, don't mind it.
    if ((video_format->coded_width >> 4) * (video_format->coded_height >> 4) > decode_caps.nMaxMBCount){

        std::ostringstream error_ss;
        error_ss << std::endl
                    << "MBCount             : " << (video_format->coded_width >> 4)*(video_format->coded_height >> 4) << std::endl
                    << "Max Supported mbcnt : " << decode_caps.nMaxMBCount << std::endl
                    << "MBCount not supported on this GPU";
        INFO_ERROR(error_ss.str().c_str());
        return n_decoded_surfaces;
    }
    // cuvidCreateDecoder() has been called before, and now there's possible config change.
    if (width_ && luma_height_ && chroma_height_) 
        return reconfig_decoder(video_format);

    // codec has been set in the constructor (for parser). Here it's set again for potential correction
    video_codec_ = video_format->codec;
    video_chroma_format_ = video_format->chroma_format;
    bit_depth_minus8_ = video_format->bit_depth_luma_minus8;
    bytes_per_pixel_ = bit_depth_minus8_ > 0 ? 2 : 1;

    // Set the output surface format same as chroma format.
    if (video_chroma_format_ == cudaVideoChromaFormat_420)
        video_surface_format_ = video_format->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    else if (video_chroma_format_ == cudaVideoChromaFormat_444)
        video_surface_format_ = video_format->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
    else if (video_chroma_format_ == cudaVideoChromaFormat_422)
        video_surface_format_ = cudaVideoSurfaceFormat_NV12;  // no 4:2:2 output format supported yet so make 420 default

    // Check if output format supported, in most cases input stream is the same as the output, but in certain cases, it may be different.
    if (!(decode_caps.nOutputFormatMask & (1 << video_surface_format_)))  // do not support current format
    {
        // Try to find one support format.
        if (decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_NV12))
            video_surface_format_ = cudaVideoSurfaceFormat_NV12;
        else if (decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_P016))
            video_surface_format_ = cudaVideoSurfaceFormat_P016;
        else if (decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444))
            video_surface_format_ = cudaVideoSurfaceFormat_YUV444;
        else if (decode_caps.nOutputFormatMask & (1 << cudaVideoSurfaceFormat_YUV444_16Bit))
            video_surface_format_ = cudaVideoSurfaceFormat_YUV444_16Bit;
        else 
            INFO_ERROR("No supported output format found");
    }
    video_format_ = *video_format;

    // To fill the cuvidCreateDecoder()
    CUVIDDECODECREATEINFO video_decode_create_info = { 0 };
    video_decode_create_info.CodecType = video_format->codec;
    video_decode_create_info.ChromaFormat = video_format->chroma_format;
    video_decode_create_info.OutputFormat = video_surface_format_;
    video_decode_create_info.bitDepthMinus8 = video_format->bit_depth_luma_minus8;
    // cudaVideoDeinterlaceMode_Weave or cudaVideoDeinterlaceMode_Bob for progressive content.
    if (video_format->progressive_sequence)
        video_decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    else  // cudaVideoDeinterlaceMode_Adaptive for interlaced content.
        video_decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    // The maximum number of output surfaces.
    video_decode_create_info.ulNumOutputSurfaces = 2;
    // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
    video_decode_create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    // Using a higher number ensures better pipelining but increases GPU memory consumption. 
    video_decode_create_info.ulNumDecodeSurfaces = n_decoded_surfaces;
    video_decode_create_info.vidLock = video_ctx_lock_;
    // Coded height and width in pixel.
    video_decode_create_info.ulWidth = video_format->coded_width;
    video_decode_create_info.ulHeight = video_format->coded_height;
    if (max_width_ < (int)video_format->coded_width)
        max_width_ = video_format->coded_width;
    if (max_height_ < (int)video_format->coded_height)
        max_height_ = video_format->coded_height;
    // The max width and height that decoder support in case of resolutiuon change.
    video_decode_create_info.ulMaxWidth = max_width_;
    video_decode_create_info.ulMaxHeight = max_height_;
    // Get the resolution of output surface.
    if (!(crop_area_.r && crop_area_.b) && !(resize_dim_.w && resize_dim_.h))
    {
        width_ = video_format->display_area.right - video_format->display_area.left;
        luma_height_ = video_format->display_area.bottom - video_format->display_area.top;
        video_decode_create_info.ulTargetWidth = video_format->coded_width;
        video_decode_create_info.ulTargetHeight = video_format->coded_height;
    } 
    else 
    {
        if (resize_dim_.w && resize_dim_.h)  // resize required. 
        {
            video_decode_create_info.display_area.left = video_format->display_area.left;
            video_decode_create_info.display_area.top = video_format->display_area.top;
            video_decode_create_info.display_area.right = video_format->display_area.right;
            video_decode_create_info.display_area.bottom = video_format->display_area.bottom;
            width_ = resize_dim_.w;
            luma_height_ = resize_dim_.h;
        }

        if (crop_area_.r && crop_area_.b)  // crop required.
        {
            video_decode_create_info.display_area.left = crop_area_.l;
            video_decode_create_info.display_area.top = crop_area_.t;
            video_decode_create_info.display_area.right = crop_area_.r;
            video_decode_create_info.display_area.bottom = crop_area_.b;
            width_ = crop_area_.r - crop_area_.l;
            luma_height_ = crop_area_.b - crop_area_.t;
        }
        video_decode_create_info.ulTargetWidth = width_;
        video_decode_create_info.ulTargetHeight = luma_height_;
    }

    chroma_height_ = static_cast<uint32_t>(luma_height_ * get_chroma_height_factor(video_surface_format_));
    n_chroma_planes_ = get_chromal_plane_cnt(video_surface_format_);
    // surface is output.
    surface_height_ = video_decode_create_info.ulTargetHeight;
    surface_width_ = video_decode_create_info.ulTargetWidth;
    display_rect_.b = video_decode_create_info.display_area.bottom;
    display_rect_.t = video_decode_create_info.display_area.top;
    display_rect_.l = video_decode_create_info.display_area.left;
    display_rect_.r = video_decode_create_info.display_area.right;

    video_info_ << "Video Decoding Params:" << std::endl
        << "\tNum Surfaces : " << video_decode_create_info.ulNumDecodeSurfaces << std::endl
        << "\tCrop         : [" << video_decode_create_info.display_area.left << ", " << video_decode_create_info.display_area.top << ", "
        << video_decode_create_info.display_area.right << ", " << video_decode_create_info.display_area.bottom << "]" << std::endl
        << "\tResize       : " << video_decode_create_info.ulTargetWidth << "x" << video_decode_create_info.ulTargetHeight << std::endl
        << "\tDeinterlace  : " << std::vector<const char *>{"Weave", "Bob", "Adaptive"}[video_decode_create_info.DeinterlaceMode];
    video_info_ << std::endl;

    cuvidCreateDecoder(&video_decoder_, &video_decode_create_info);
    return n_decoded_surfaces;

}

/// @brief Call back function when stream is ready to decode.
/// @param pic_params 
/// @return (int): 0 on failed, 1 on successed.
int NVDecoderImpl::handle_picture_decode(CUVIDPICPARAMS *pic_params) 
{
    if (!video_decoder_)
    {
        INFO_ERROR("Decoder not initialized.");
        return 0;
    }
    // pic_num_in_decode_order_[pic_params->CurrPicIdx] = n_decoded_pic_cnt_++;
    // kick off decoding.
    cuvidDecodePicture(video_decoder_, pic_params);  
    return 1;
}

/// @brief Call back function when one frame is ready to display.
/// @param display_info 
/// @return 
int NVDecoderImpl::handle_picture_display(CUVIDPARSERDISPINFO *display_info) 
{
    CUVIDPROCPARAMS video_processing_params = {};
    video_processing_params.progressive_frame = display_info->progressive_frame;
    video_processing_params.second_field = display_info->repeat_first_field + 1;
    video_processing_params.top_field_first = display_info->top_field_first;
    video_processing_params.unpaired_field = display_info->repeat_first_field < 0;
    video_processing_params.output_stream = cuvid_stream_;

    CUdeviceptr src_frame = 0;  // cuda device pointer.
    unsigned int src_pitch = 0;  // the pitch.
    // User need to call cuvidMapVideoFrame() to get CUDA device pointer and pitch of the output surface.
    // It's a blocking call as it waits for decoding to complete
    CHECK_CUDA_DRIVER(cuvidMapVideoFrame(video_decoder_, display_info->picture_index, &src_frame, &src_pitch, &video_processing_params));

    // To fill cuvidGetDecodeStatus()
    CUVIDGETDECODESTATUS decode_status = {};
    CUresult result = cuvidGetDecodeStatus(video_decoder_, display_info->picture_index, &decode_status);
    if (result == CUDA_SUCCESS && (decode_status.decodeStatus == cuvidDecodeStatus_Error || decode_status.decodeStatus == cuvidDecodeStatus_Error_Concealed))
        printf("Decode Error occurred for picture\n");  //, pic_num_in_decode_order_[display_info->picture_index]);

    // Allocate buffer for memory copy.
    uint8_t *decoded_frame = nullptr;
    {
        // std::lock_guard<std::mutex> lock(frame_mtx_);
        size_t bytes = get_frame_size();
        MemoryType mem_ty = MemoryType::MEM_PAGE_LOCKED;
        ++n_decoded_frames_;
        if (use_device_frame_)
        {
            mem_ty = MemoryType::MEM_DEVICE;
            if (device_frame_pitched_)
            {
                CHECK_CUDA_RUNTIME(
                    cudaMallocPitch(
                        &decoded_frame, 
                        &n_device_frame_pitched_, 
                        width_ * bytes_per_pixel_, 
                        luma_height_ + (chroma_height_ * n_chroma_planes_)
                    )
                );
                bytes = n_device_frame_pitched_ * (luma_height_ + (chroma_height_ * n_chroma_planes_));
            }
            else
                CHECK_CUDA_RUNTIME(cudaMalloc(&decoded_frame, bytes));
        }
        else
            CHECK_CUDA_RUNTIME(cudaMallocHost(&decoded_frame, bytes));
        Frame frame(
            decoded_frame, 
            bytes, 
            mem_ty, 
            width_, 
            luma_height_, 
            nvformat_to_avpixfomat(video_surface_format_), 
            display_info->timestamp, 
            mem_ty == MemoryType::MEM_DEVICE ? device_id_ : -1
        );
        if (frame_cache_.full())
            frame_cache_.pop();
        frame_cache_.push(frame);
    }
    cudaMemcpyKind transfer_kind = use_device_frame_ ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost; 
    uint32_t dst_pitch = device_frame_pitched_ ? n_device_frame_pitched_ : width_ * bytes_per_pixel_;
    CHECK_CUDA_RUNTIME(
        cudaMemcpy2DAsync(
            decoded_frame, 
            dst_pitch, 
            (uint8_t *)src_frame, 
            src_pitch, 
            width_ * bytes_per_pixel_, 
            luma_height_, 
            transfer_kind, 
            cuvid_stream_
        )
    );

    // Copy first chroma plane.
    uint8_t *src_chroma = reinterpret_cast<uint8_t *>(src_frame) + src_pitch * surface_height_;
    uint8_t *dst_chroma = decoded_frame + dst_pitch * luma_height_;
    CHECK_CUDA_RUNTIME(
        cudaMemcpy2DAsync(
            dst_chroma, 
            dst_pitch,
            src_chroma,
            src_pitch,
            width_ * bytes_per_pixel_,
            chroma_height_,
            transfer_kind,
            cuvid_stream_
        )
    );
    if (n_chroma_planes_ == 2)
    {
        src_chroma += src_pitch * surface_height_;
        dst_chroma += dst_pitch * luma_height_;
        CHECK_CUDA_RUNTIME(
            cudaMemcpy2DAsync(
                dst_chroma, 
                dst_pitch,
                src_chroma,
                src_pitch,
                width_ * bytes_per_pixel_,
                chroma_height_,
                transfer_kind,
                cuvid_stream_
            )
        );
    }
    CHECK_CUDA_RUNTIME(cudaStreamSynchronize(cuvid_stream_));
    cuvidUnmapVideoFrame(video_decoder_, src_frame);
    return 1;
}

int NVDecoderImpl::reconfig_decoder(CUVIDEOFORMAT *video_format)
{
    if (video_format->bit_depth_luma_minus8 != video_format_.bit_depth_luma_minus8 || video_format->bit_depth_chroma_minus8 != video_format_.bit_depth_chroma_minus8)
        INFO_ERROR("Reconfigure Not supported for bit depth change");

    if (video_format->chroma_format != video_format_.chroma_format)
        INFO_ERROR("Reconfigure Not supported for chroma format change");

    bool decode_res_change = !(video_format->coded_width == video_format_.coded_width && video_format->coded_height == video_format_.coded_height);
    bool display_rect_change = !(video_format->display_area.bottom == video_format_.display_area.bottom && video_format->display_area.top == video_format_.display_area.top \
        && video_format->display_area.left == video_format_.display_area.left && video_format->display_area.right == video_format_.display_area.right);

    int n_decode_surfaces = video_format->min_num_decode_surfaces;

    if ((video_format->coded_width > max_width_) || (video_format->coded_height > max_height_)) 
    {
        // For VP9, let driver  handle the change if new width/height > maxwidth/maxheight
        if ((video_codec_ != cudaVideoCodec_VP9) || reconfig_external_)
            INFO_ERROR("Reconfigure Not supported when width/height > maxwidth/maxheight");
        return 1;
    }

    if (!decode_res_change && !reconfig_external_pp_change_) {
        // if the coded_width/coded_height hasn't changed but display resolution has changed, then need to update width/height for
        // correct output without cropping. Example : 1920x1080 vs 1920x1088
        if (display_rect_change)
        {
            width_ = video_format->display_area.right - video_format->display_area.left;
            luma_height_ = video_format->display_area.bottom - video_format->display_area.top;
            chroma_height_ = int(luma_height_ * get_chroma_height_factor(video_surface_format_));
            n_chroma_planes_ = get_chromal_plane_cnt(video_surface_format_);
        }

        // no need for reconfigureDecoder(). Just return
        return 1;
    }

    CUVIDRECONFIGUREDECODERINFO reconfig_params = { 0 };

    reconfig_params.ulWidth = video_format_.coded_width = video_format->coded_width;
    reconfig_params.ulHeight = video_format_.coded_height = video_format->coded_height;

    // Dont change display rect and get scaled output from decoder. This will help display app to present apps smoothly
    reconfig_params.display_area.bottom = display_rect_.b;
    reconfig_params.display_area.top = display_rect_.t;
    reconfig_params.display_area.left = display_rect_.l;
    reconfig_params.display_area.right = display_rect_.r;
    reconfig_params.ulTargetWidth = surface_width_;
    reconfig_params.ulTargetHeight = surface_height_;

    // If external reconfigure is called along with resolution change even if post processing params is not changed,
    // do full reconfigure params update
    if ((reconfig_external_ && decode_res_change) || reconfig_external_pp_change_) {
        // update display rect and target resolution if requested explicitely
        reconfig_external_ = false;
        reconfig_external_pp_change_ = false;
        video_format_ = *video_format;
        if (!(crop_area_.r && crop_area_.b) && !(resize_dim_.w && resize_dim_.h)) {
            width_ = video_format->display_area.right - video_format->display_area.left;
            luma_height_ = video_format->display_area.bottom - video_format->display_area.top;
            reconfig_params.ulTargetWidth = video_format->coded_width;
            reconfig_params.ulTargetHeight = video_format->coded_height;
        }
        else {
            if (resize_dim_.w && resize_dim_.h) {
                reconfig_params.display_area.left = video_format->display_area.left;
                reconfig_params.display_area.top = video_format->display_area.top;
                reconfig_params.display_area.right = video_format->display_area.right;
                reconfig_params.display_area.bottom = video_format->display_area.bottom;
                width_ = resize_dim_.w;
                luma_height_ = resize_dim_.h;
            }

            if (crop_area_.r && crop_area_.b) {
                reconfig_params.display_area.left = crop_area_.l;
                reconfig_params.display_area.top = crop_area_.t;
                reconfig_params.display_area.right = crop_area_.r;
                reconfig_params.display_area.bottom = crop_area_.b;
                width_ = crop_area_.r - crop_area_.l;
                luma_height_ = crop_area_.b - crop_area_.t;
            }
            reconfig_params.ulTargetWidth = width_;
            reconfig_params.ulTargetHeight = luma_height_;
        }

        chroma_height_ = int(luma_height_ * get_chroma_height_factor(video_surface_format_));
        n_chroma_planes_ = get_chromal_plane_cnt(video_surface_format_);
        surface_height_ = reconfig_params.ulTargetHeight;
        surface_width_ = reconfig_params.ulTargetWidth;
        display_rect_.b = reconfig_params.display_area.bottom;
        display_rect_.t = reconfig_params.display_area.top;
        display_rect_.l = reconfig_params.display_area.left;
        display_rect_.r = reconfig_params.display_area.right;
    }

    reconfig_params.ulNumDecodeSurfaces = n_decode_surfaces;
    // cuCtxPushCurrent(cu_ctx_);
    cuvidReconfigureDecoder(video_decoder_, &reconfig_params);
    // cuCtxPopCurrent(nullptr);

    return n_decode_surfaces;
}

int NVDecoderImpl::set_reconfig_params(const Rect *crop_area, const Dim *resize_dim)
{
    reconfig_external_ = true;
    reconfig_external_pp_change_ = false;
    if (crop_area)
    {
        if (!((crop_area->t == crop_area_.t) && (crop_area->l == crop_area_.l) &&
            (crop_area->b == crop_area_.b) && (crop_area->r == crop_area_.r)))
        {
            reconfig_external_pp_change_ = true;
            crop_area_ = *crop_area;
        }
    }
    if (resize_dim)
    {
        if (!((resize_dim->w == resize_dim_.w) && (resize_dim->h == resize_dim_.h)))
        {
            reconfig_external_pp_change_ = true;
            resize_dim_ = *resize_dim;
        }
    }
    frame_cache_.clear();
    return 1;
}

std::shared_ptr<NVDecoder> create_hardware_decoder(
    cudaVideoCodec codec, 
    int device_id,
    bool use_device_frame, 
    bool low_latency, 
    bool device_frame_pitched,
    const Rect *crop_area,
    const Dim *resize_dim,
    int max_width,
    int max_height,
    uint32_t clock_rate,
    int cache_size
) {
    std::shared_ptr<NVDecoderImpl> instance(new NVDecoderImpl());
    if (!instance->create(codec, device_id, use_device_frame, low_latency, device_frame_pitched, crop_area, resize_dim, max_width, max_height, clock_rate, cache_size))
    {
        instance.reset();
        return nullptr;
    }
    return instance;
}
