#ifndef CONTIGUOUS_CUH
#define CONTIGUOUS_CUH
#include <mix_memory/mix_mat.hpp>

// remap index from src to dst.
__host__ __device__ size_t remap_index(
    size_t src_index, 
    int n_src_strides, size_t *src_strides,
    int n_dst_strides, size_t *dst_strides
);

__host__ void contiguous_async(
    void *dst, void *src, 
    const int n_strides, size_t *strides, 
    const int n_dims, int *dims,
    size_t numel,
    tinycv::DataType dtype,
    cudaStream_t stream = nullptr,
    bool sync = true
);

#endif  // CONTIGUOUS_CUH