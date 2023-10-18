#ifndef CHECK_H
#define CHECK_H

#include "ilogger.hpp"
#include <cuda_runtime.h>

#define CHECK_FFMPEG(op) check_ffmpeg_((op), #op, __FILE__, __LINE__)
#define CHECK_CUDA_RUNTIME(op) check_cuda_runtime_((op), #op, __FILE__, __LINE__)
#define CHECK_CUDA_DRIVER(op) check_cuda_driver_((op), #op, __FILE__, __LINE__)
                               
#define CHECK_CUDA_KERNEL(...)                                                                       \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        INFO_ERROR("launch failed: %s", cudaGetErrorString(cudaStatus));                             \
    }} while(0);

bool check_ffmpeg_(int re, const char *op, const char *file_name, int line);
bool check_cuda_runtime_(int re, const char *op, const char *file_name, int line);
bool check_cuda_driver_(int re, const char *op, const char *file_name, int line);


#endif  // CHECK_H