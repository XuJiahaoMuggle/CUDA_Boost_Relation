#include "utils.h"
#include <cuda_runtime.h>

namespace cuda_tools
{
    int get_device_id(int device_id)
    {
        if (check_device_id(device_id))
            return device_id;
        CHECK_CUDA_RUNTIME(cudaGetDevice(&device_id));
        return device_id;
    }

    bool check_device_id(int device_id)
    {
        int cnt;
        CHECK_CUDA_RUNTIME(cudaGetDeviceCount(&cnt));
        if (cnt <= device_id || device_id < 0)
            return false;
        return true;
    }

    AutoExchangeDevice::AutoExchangeDevice(int device_id)
    {
        if (!check_device_id(device_id))
            return;
        CHECK_CUDA_RUNTIME(cudaGetDevice(&old_device_id_));
        if (old_device_id_ != device_id)
        { 
            CHECK_CUDA_RUNTIME(cudaSetDevice(device_id));
        }
    }

    AutoExchangeDevice::~AutoExchangeDevice()
    {
        if (old_device_id_ != -1)
        {
            CHECK_CUDA_RUNTIME(cudaSetDevice(old_device_id_));
        }
    }

    CUcontext create_cuda_ctx(int flags, int device_id)
    {
        CUcontext cu_ctx;
        CHECK_CUDA_DRIVER(cuCtxCreate(&cu_ctx, flags, device_id));
        return cu_ctx;
    }

};