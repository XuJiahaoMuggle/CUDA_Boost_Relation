#include "base_memory.hpp"
#include <utils/utils.h>

namespace tinycv
{
    BaseMemory::BaseMemory(int device_id)
    : device_id_(cuda_tools::get_device_id(device_id))
    {}
    
    BaseMemory::BaseMemory(void *p_host, size_t host_size, void *p_device, size_t device_size)
    {
        ref_data(p_host, host_size, p_device, device_size);
    }

    void BaseMemory::release_host()
    {
        if (host_owner_)
        {
            cuda_tools::AutoExchangeDevice device_exchange(device_id_);
            CHECK_CUDA_RUNTIME(cudaFreeHost(host_data_));
            host_data_size_ = 0;
            host_data_ = nullptr;
        }
    }

    void BaseMemory::release_device()
    {
        if (device_owner_)
        {
            cuda_tools::AutoExchangeDevice device_exchange(device_id_);
            CHECK_CUDA_RUNTIME(cudaFree(device_data_));
            device_data_size_ = 0;
            device_data_ = nullptr;            
        }
    }
    
    void BaseMemory::release_all()
    {
        release_host();
        release_device();
    }

    void BaseMemory::ref_data(void *p_host, size_t host_size, void *p_device, size_t device_size)
    {
        release_all();
        host_data_ = p_host && host_size > 0 ? p_host : nullptr;
        host_data_size_ = host_data_ ? host_size : 0;

        device_data_ = p_device && device_size > 0 ? p_device : nullptr;
        device_data_size_ = device_data_ ? device_size : 0;

        host_owner_ = !(p_host && host_size > 0);
        device_owner_ = !(p_device && device_size > 0);
        CHECK_CUDA_RUNTIME(cudaGetDevice(&device_id_));  // switch device to ref data's device.
    }

    void *BaseMemory::host(size_t size)
    {
        if (host_data_size_ < size)
        {
            release_host();
            cuda_tools::AutoExchangeDevice device_exchange(device_id_);
            CHECK_CUDA_RUNTIME(cudaMallocHost(&host_data_, size));
            host_data_size_ = size;
            host_owner_ = true;
        }
        return host_data_;
    }

    void *BaseMemory::device(size_t size)
    {
        if (device_data_size_ < size)
        {
            release_device();
            cuda_tools::AutoExchangeDevice device_exchange(device_id_);
            CHECK_CUDA_RUNTIME(cudaMalloc(&device_data_, size));
            device_data_size_ = size;
            device_owner_ = true;
        }
        return device_data_;
    }
};