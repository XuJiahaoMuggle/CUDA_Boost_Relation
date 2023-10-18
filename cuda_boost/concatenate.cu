#include "concatenate.cuh"
#include <cassert>
#include <mix_memory/base_memory.hpp>
#include <utils/utils.h>

namespace tinycv
{
    __host__ bool compare_dims(const std::vector<int> &dims0, const std::vector<int> &dims1, int except_index)
    {
        if (dims0.size() != dims1.size() || except_index >= dims0.size())
            return false;
        for (int i = 0; i < dims0.size(); ++i)
        {
            if (except_index != i && dims0[i] != dims1[i])
                return false;
        }
        return true;
    }
    
    MixMat concatenate(const std::vector<MixMat> &mats, int index, cudaStream_t stream, bool sync)
    {
        return concatenate(mats.size(), mats.data(), index, stream, sync);
    }

    template <typename dtype>
    __global__ void concatenate_memory_move_kernel(
        void *dst,  int n_dims, size_t *dst_strides, size_t bytes_dst,
        int concat_dim, int *cumsum, int n_srcs, void **srcs, size_t **srcs_strides
    ) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ size_t dst_strides_shared[10];
        __shared__ size_t cumsum_shared[10];
        extern __shared__ size_t src_strides_shared[];
        if (threadIdx.x < n_dims)
        {
            cumsum_shared[threadIdx.x] = cumsum[threadIdx.x];
            dst_strides_shared[threadIdx.x] = dst_strides[threadIdx.x];
            for (int i = 0; i < n_srcs; ++i)
                src_strides_shared[i * n_dims + threadIdx.x] = srcs_strides[i][threadIdx.x];
        }
        __syncthreads();
        for (size_t i = idx; i < bytes_dst; i += gridDim.x * blockDim.x)
        {
            int indices[4] = {0};
            size_t rest = i;
            for (int j = 0; j < n_dims; ++j)
            {
                indices[j] = rest / dst_strides_shared[j];
                rest %= dst_strides_shared[j];
            }
            
            int srcs_idx = 0;
            for (; srcs_idx < n_srcs; ++srcs_idx)
            {
                if (indices[concat_dim] <= cumsum_shared[srcs_idx] - 1)
                {
                    indices[concat_dim] = cumsum_shared[srcs_idx] - 1 - indices[concat_dim];
                    break;
                }
            }

            size_t src_idx = 0;
            for (int j = 0; j < n_dims; ++j)
                src_idx += src_strides_shared[srcs_idx * n_dims + j] * indices[j];
         
            *((dtype *)dst + i) = *(*((dtype **)srcs + srcs_idx) + src_idx);
        }
    }
    
    static void assert_concatenate(int n_mats, const MixMat *mats, int index)
    {
        const std::vector<int> &ref_shape = mats[0].dims();
        const int ref_n_dims = mats[0].n_dims();
        assert(n_mats > 0 && mats != nullptr && index < ref_n_dims);
        for (int i = 1; i < n_mats; ++i)
        {
            assert(ref_n_dims == mats[i].n_dims());
            for (int j = 0; j < ref_n_dims; ++j)
            {
                if (j != index)
                    assert(ref_shape[j] == mats[i].dims()[j]);
            }
        }
    }
    
    MixMat concatenate(int n_mats, const MixMat *mats, int index, cudaStream_t stream, bool sync)
    {
        assert_concatenate(n_mats, mats, index);
        int n_dims = mats[0].n_dims();  // number of dims.
        std::vector<int> cumsum_dims(n_dims, 0);  // cumsum of concat index.
        cumsum_dims[0] = mats[0].dims()[index];
        for (int i = 1; i < n_mats; ++i)
            cumsum_dims[i] = mats[i].dims()[index] + cumsum_dims[i - 1];
        int *p_cumsum = nullptr;  // remember to free
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_cumsum, cumsum_dims.size() * sizeof(int)));
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(p_cumsum, cumsum_dims.data(), cumsum_dims.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

        size_t **pp_strides = nullptr;  // remember to free.
        size_t bytes_per_stride = sizeof(size_t) * n_dims;
        CHECK_CUDA_RUNTIME(cudaMalloc(&pp_strides, sizeof(size_t *) * n_mats));  
        std::vector<size_t *> p_strides(n_mats, nullptr);  // remember to free
        for (int i = 0; i < n_mats; ++i)
        {
            CHECK_CUDA_RUNTIME(cudaMalloc(&p_strides[i], bytes_per_stride));
            CHECK_CUDA_RUNTIME(cudaMemcpyAsync(p_strides[i], mats[i].strides().data(), bytes_per_stride, cudaMemcpyHostToDevice, stream));
        }
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(pp_strides, p_strides.data(), sizeof(size_t *) * n_mats, cudaMemcpyHostToDevice, stream));

        void **pp_src = nullptr;  // remember to free
        CHECK_CUDA_RUNTIME(cudaMalloc(&pp_src, sizeof(void *) * n_mats));

        std::vector<void *> p_src(n_mats, nullptr);  // no need to free.
        for (int i = 0; i < n_mats; ++i)
            p_src[i] = mats[i].device();
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(pp_src, p_src.data(), sizeof(void *) * n_mats, cudaMemcpyHostToDevice, stream));
        
        // get dst index
        int dst_dim = 0;
        for (int i = 0; i < n_mats; ++i)
            dst_dim += mats[i].dims()[index];
        std::vector<int> dst_dims = mats[0].dims();
        dst_dims[index] = dst_dim;

        MixMat dst(dst_dims, mats[0].dtype());
        size_t *p_dst_strides = nullptr;  // remember to free
        const std::vector<size_t> &dst_strides = dst.strides();
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_dst_strides, dst_strides.size() * sizeof(size_t)));
        CHECK_CUDA_RUNTIME(cudaMemcpyAsync(p_dst_strides, dst_strides.data(),  dst_strides.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream));

        CHECK_CUDA_KERNEL(
        concatenate_memory_move_kernel<uint8_t><<<256, 32, n_mats * n_dims * sizeof(size_t), stream>>>(
            dst.device(), n_dims, p_dst_strides, dst.bytes(), 
            index, p_cumsum, n_mats, pp_src, pp_strides
        ));

        CHECK_CUDA_RUNTIME(cudaFree(p_dst_strides));
        CHECK_CUDA_RUNTIME(cudaFree(pp_src));
        for (int i = 0; i < n_mats; ++i)
            CHECK_CUDA_RUNTIME(cudaFree(p_strides[i]));
        CHECK_CUDA_RUNTIME(cudaFree(pp_strides));
        CHECK_CUDA_RUNTIME(cudaFree(p_cumsum));
        
        if (sync)
            CHECK_CUDA_RUNTIME(cudaStreamSynchronize(stream));
        return dst;
    }

};