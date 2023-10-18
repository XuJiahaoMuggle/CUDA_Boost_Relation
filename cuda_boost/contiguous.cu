#include "contiguous.cuh"
#include <cuda_fp16.h>
#include <utils/utils.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
using one_byte_t = uint8_t;

#define BLOCK_SIZE 256
#define GRID_SIZE 32

// remap index from src to dst.
__device__ size_t remap_index(
    size_t src_index, 
    int n_src_strides, size_t *src_strides,
    int n_dst_strides, size_t *dst_strides
) {
    size_t dst_index = 0;
    for (int i = 0; i < n_src_strides && src_index > 0; ++i)
    {
        // src and dst share the same indices.
        dst_index += src_index / src_strides[i] * dst_strides[i];
        src_index %= src_strides[i];
    }
    return dst_index;
}

template <typename dtype>
__global__ void contiguous_kernel(
    dtype *dst, dtype *src, size_t numel,
    int n_dst_strides, size_t *dst_strides,
    int n_src_strides, size_t *src_strides
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__  size_t dst_strides_shared[10];
    __shared__  size_t src_strides_shared[10];

    if (threadIdx.x < n_dst_strides)
    {
        dst_strides_shared[threadIdx.x] = dst_strides[threadIdx.x];
        src_strides_shared[threadIdx.x] = src_strides[threadIdx.x];
    }
    __syncthreads();
    // grid loop.
    for (size_t dst_idx = (size_t)idx; dst_idx < numel; dst_idx += blockDim.x * gridDim.x)
    {
        size_t src_idx = remap_index(dst_idx, n_dst_strides, dst_strides_shared, n_src_strides, src_strides_shared);
        *(dst + dst_idx) = *(src + src_idx);
    }
}

__host__ void contiguous_async(
    void *dst, void *src, 
    const int n_strides, size_t *strides, 
    const int n_dims, int *dims,
    size_t numel,
    tinycv::DataType dtype,
    cudaStream_t stream,
    bool sync
)
{
    assert(dst != src && n_dims == n_strides);
    size_t *src_strides, *dst_strides;
    CHECK_CUDA_RUNTIME(cudaMalloc(&src_strides, n_strides * sizeof(size_t)));
    CHECK_CUDA_RUNTIME(cudaMemcpyAsync(src_strides, strides, n_strides * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    // contiguous strides.
    for (int i = n_strides - 1; i >= 0; --i)
    {
        if (i == n_strides - 1)
            strides[i] = 1;
        else
            strides[i] = strides[i + 1] * dims[i + 1];
    }
    CHECK_CUDA_RUNTIME(cudaMalloc(&dst_strides, n_strides * sizeof(size_t)));
    CHECK_CUDA_RUNTIME(cudaMemcpyAsync(dst_strides, strides, n_strides * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    switch (dtype)
    {
    case tinycv::DataType::FLOAT32:
        CHECK_CUDA_KERNEL(
            contiguous_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                static_cast<float *>(dst), static_cast<float *>(src), 
                numel, n_strides, dst_strides, n_strides, src_strides
            )
        );
        break;
    case tinycv::DataType::FLOAT16:
        CHECK_CUDA_KERNEL(
            contiguous_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                static_cast<tinycv::float16 *>(dst), static_cast<tinycv::float16 *>(src), 
                numel, n_strides, dst_strides, n_strides, src_strides
            )
        );
        break;
    case tinycv::DataType::UINT8:
        CHECK_CUDA_KERNEL(
            contiguous_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                static_cast<uint8_t *>(dst), static_cast<uint8_t *>(src), 
                numel, n_strides, dst_strides, n_strides, src_strides
            )
        );
        break;
    case tinycv::DataType::INT8:
        CHECK_CUDA_KERNEL(
            contiguous_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                static_cast<int8_t *>(dst), static_cast<int8_t *>(src), 
                numel, n_strides, dst_strides, n_strides, src_strides
            )
        );
        break;
    case tinycv::DataType::INT32:
        CHECK_CUDA_KERNEL(
            contiguous_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
                static_cast<int32_t *>(dst), static_cast<int32_t *>(src), 
                numel, n_strides, dst_strides, n_strides, src_strides
            )
        );
        break;
    default:
        INFO_FATAL("Invalid data type %d when contiguous.", static_cast<int>(dtype));
        break;
    }
    if (sync)
        CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream));
}
