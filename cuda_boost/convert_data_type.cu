#include "convert_data_type.cuh"
#include <utils/utils.h>

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 32

template <typename dst_dtype, typename src_dtype>
__global__ void convert_dtype_kernel(dst_dtype *new_data, src_dtype *old_data, size_t size)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // grid loop
    for (;idx < size; idx += blockDim.x * gridDim.x)
    {
        *(new_data + idx) = *(old_data + idx);
    }
}

template <>
__global__ void convert_dtype_kernel(float *new_data, tinycv::float16 *old_data, size_t size)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // grid loop
    for (;idx < size; idx += blockDim.x * gridDim.x)
    {
        *(new_data + idx) = __half2float(*((__half *)old_data + idx));
    }
}

template <>
__global__ void convert_dtype_kernel(int32_t *new_data, tinycv::float16 *old_data, size_t size)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // grid loop
    for (;idx < size; idx += blockDim.x * gridDim.x)
    {
        *(new_data + idx) = __half2int_rn(*((__half *)old_data + idx));
    }
}

template <>
__global__ void convert_dtype_kernel(tinycv::float16 *new_data, float *old_data, size_t size)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // grid loop
    for (;idx < size; idx += blockDim.x * gridDim.x)
    {
        *((__half *)new_data + idx) = __float2half(*(old_data + idx));
    }
}

template <>
__global__ void convert_dtype_kernel(tinycv::float16 *new_data, int32_t *old_data, size_t size)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // grid loop
    for (;idx < size; idx += blockDim.x * gridDim.x)
    {
        *((__half *)new_data + idx) = __int2half_rn(*(old_data + idx));
    }
}

__host__ void convert_data_type_async(
    void *new_data, tinycv::DataType dst_dtype, 
    void *old_data, tinycv::DataType src_dtype, 
    size_t size, cudaStream_t stream, bool sync
)
{
    switch (dst_dtype)
    {
    case tinycv::DataType::FLOAT32:  // support float32
        if (src_dtype == tinycv::DataType::INT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<float *>(new_data), static_cast<int32_t *>(old_data), size);
        else if (src_dtype == tinycv::DataType::UINT8)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<float *>(new_data), static_cast<uint8_t *>(old_data), size);
        else if (src_dtype == tinycv::DataType::INT8)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<float *>(new_data), static_cast<int8_t *>(old_data), size);
        else if (src_dtype == tinycv::DataType::FLOAT16)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<float *>(new_data), static_cast<tinycv::float16 *>(old_data), size);
        else
            INFO_FATAL("Unsupported format conversion from %s to %s.", tinycv::data_type_str(src_dtype), tinycv::data_type_str(dst_dtype));
        break;

    case tinycv::DataType::INT8:
        if (src_dtype == tinycv::DataType::FLOAT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<int8_t *>(new_data), static_cast<float *>(old_data), size);
        else if (src_dtype == tinycv::DataType::INT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<int8_t *>(new_data), static_cast<int32_t *>(old_data), size);
        else if (src_dtype == tinycv::DataType::UINT8)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<int8_t *>(new_data), static_cast<uint8_t *>(old_data), size);
        else
            INFO_FATAL("Unsupported format conversion from %s to %s.", tinycv::data_type_str(src_dtype), tinycv::data_type_str(dst_dtype));
        break;

    case tinycv::DataType::UINT8:
        if (src_dtype == tinycv::DataType::FLOAT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<uint8_t *>(new_data), static_cast<float *>(old_data), size);
        else if (src_dtype == tinycv::DataType::INT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<uint8_t *>(new_data), static_cast<int32_t *>(old_data), size);
        else if (src_dtype == tinycv::DataType::INT8)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<uint8_t *>(new_data), static_cast<int8_t *>(old_data), size); 
        else
            INFO_FATAL("Unsupported format conversion from %s to %s.", tinycv::data_type_str(src_dtype), tinycv::data_type_str(dst_dtype));
        break;

    case tinycv::DataType::INT32:
        if (src_dtype == tinycv::DataType::FLOAT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<int32_t *>(new_data), static_cast<float *>(old_data), size);
        else if (src_dtype == tinycv::DataType::UINT8)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<int32_t *>(new_data), static_cast<uint8_t *>(old_data), size);
        else if (src_dtype == tinycv::DataType::INT8)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<int32_t *>(new_data), static_cast<int8_t *>(old_data), size);
        else if (src_dtype == tinycv::DataType::FLOAT16)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<int32_t *>(new_data), static_cast<tinycv::float16 *>(old_data), size);
        else
            INFO_FATAL("Unsupported format conversion from %s to %s.", tinycv::data_type_str(src_dtype), tinycv::data_type_str(dst_dtype)); 
        break;

    case tinycv::DataType::FLOAT16:
        if (src_dtype == tinycv::DataType::FLOAT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<tinycv::float16 *>(new_data), static_cast<float *>(old_data), size);
        else if (src_dtype == tinycv::DataType::INT32)
            convert_dtype_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(static_cast<tinycv::float16 *>(new_data), static_cast<int32_t *>(old_data), size);             
        else
            INFO_FATAL("Unsupported format conversion from %s to %s.", tinycv::data_type_str(src_dtype), tinycv::data_type_str(dst_dtype));
        break ;

    default:
        INFO_FATAL("Unsupported format conversion from %s to %s.", tinycv::data_type_str(src_dtype), tinycv::data_type_str(dst_dtype));
    }
    if (sync)
        CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream));
}