#include "check.h"
#include "ilogger.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif 
#include <libavcodec/avcodec.h>
#ifdef __cplusplus
}
#endif

bool check_ffmpeg_(int re, const char *op, const char *file_name, int line)
{
    if (re < 0)
    {
        char buffer[1024] = {0};
        av_strerror(re, buffer, sizeof(buffer) - 1);
        INFO_ERROR(
            "FFmpeg Api Error: %s # %s, error number = %d in file %s:%d",
            op,
            buffer,
            re,
            file_name,
            line
        );
        return false;
    }
    return true;
}

bool check_cuda_runtime_(int re, const char *op, const char *file_name, int line)
{
    if ((cudaError_t)re != cudaSuccess)
    {
        INFO_ERROR(
            "CUDA Runtime Error: %s # %s, code = %s [%d] in file %s:%d",
            op,  // the name of error opeartion
            cudaGetErrorString((cudaError_t)re),  // error string
            cudaGetErrorName((cudaError_t)re),  // error coed name 
            re,  // the number of error code 
            file_name,  // file name 
            line  // file line
        );
        return false;
    }
    return true;
}

bool check_cuda_driver_(int re, const char *op, const char *file_name, int line)
{
    if (re != CUDA_SUCCESS)
    {
        const char *err_msg = nullptr;
        const char *err_name = nullptr;
        cuGetErrorString((CUresult)re, &err_msg);  
        cuGetErrorName((CUresult)re, &err_name); 
        INFO_ERROR(
            "CUDA Driver Error: %s # %s, code = %s [%d] in file %s:%d",
            op,  // the name of error opeartion
            err_msg,  // error string
            err_name,  // error coed name 
            re,  // the number of error code 
            file_name,  // file name 
            line  // file line
        );
        return false;
    }
    return true;
}
