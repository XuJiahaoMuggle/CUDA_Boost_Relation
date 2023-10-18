#include "interpolation.cuh"
#include <cstdint>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <utils/utils.h>
#include <mix_memory/mix_mat.hpp>
#include <cstdio>

namespace tinycv
{
    // affine project
    __host__ __device__ void affine_poject(const float *trans, int x, int y, float *project_x, float *project_y)
    {
        *project_x = trans[0] * x + trans[1] * y + trans[2];
        *project_y = trans[3] * x + trans[4] * y + trans[5];
    }

    // retval in [low, high]
    template <typename dtype>
    static __inline__ __device__ dtype limit(dtype value, dtype low, dtype high)
    {
        return value < low ? low : (value > high ? high : value);
    }  

    // inverse affine trans
    __host__ __device__ void inverse_affine_trans(const float *original_mat, float *inverse_mat)
    {
        float ele_0_0 = original_mat[0], ele_0_1 = original_mat[1], ele_0_2 = original_mat[2];
        float ele_1_0 = original_mat[3], ele_1_1 = original_mat[4], ele_1_2 = original_mat[5];

        float det_original_mat = ele_0_0 * ele_1_1 - ele_0_1 * ele_1_0;
        det_original_mat = det_original_mat != 0 ? 1 / det_original_mat : 0;

        inverse_mat[0] = ele_1_1 * det_original_mat;
        inverse_mat[1] = -ele_0_1 * det_original_mat;
        inverse_mat[2] = (ele_0_1 * ele_1_2 - ele_0_2 * ele_1_1) * det_original_mat;
        inverse_mat[3] = -ele_1_0 * det_original_mat;
        inverse_mat[4] = ele_0_0 * det_original_mat;
        inverse_mat[5] = (ele_1_0 * ele_0_2 - ele_0_0 * ele_1_2) * det_original_mat;
    }

    // implement 2d interpolation.
    template <typename dtype>
    __global__ void nearest_interpolate_2d_kernel(
        dtype *dst, int dst_height, int dst_width, size_t *dst_strides,  
        dtype *src, int src_height, int src_width, size_t *src_strides,  
        int n_batches, int n_channels, int n_strides, const float *trans,
        dtype fill_value = 127
    )
    {
        // we just make a indices in [N, C, H, W] format
        int indices[4] = {
            (int)blockIdx.z / n_channels,  // n index 
            (int)blockIdx.z % n_channels,  // c index
            (int)(threadIdx.y + blockDim.y * blockIdx.y),  // h index 
            (int)(threadIdx.x + blockDim.x * blockIdx.x)  // w index
        };
        // out of range.
        if (
            indices[0] >= n_batches ||
            indices[1] >= n_channels || 
            indices[2] >= dst_height || 
            indices[3] >= dst_width
        )
            return;
        // get the offset of dst.
        size_t dst_idx = 0;
        for (int i = -1; i >= -1 * n_strides; --i)
            dst_idx += indices[i + 4] * dst_strides[i + n_strides];  // implicit to size_t

        // map the coordinate of src.
        float x_src_idx, y_src_idx;
        affine_poject(trans, indices[3], indices[2], &x_src_idx, &y_src_idx);

        // memory friendly assign.
        indices[3] = (int)x_src_idx;  
        indices[2] = (int)y_src_idx;
        if (x_src_idx >= src_width || x_src_idx < 0 || y_src_idx >= src_height || y_src_idx < 0)
        {
            // *((dtype *)dst + dst_idx) = fill_value;
            *(dst + dst_idx) = fill_value;
            return;
        }
        size_t src_idx = 0;
        for (int i = -1; i >= -1 * n_strides; --i)
            src_idx += indices[i + 4] * src_strides[i + n_strides]; // implicit to size_t
        // *((dtype *)dst + dst_idx) = *((dtype *)src + src_idx);
        *(dst + dst_idx) = *(src + src_idx);
    }

    template <typename dtype>
    __global__ void bilinear_interpolate_2d_kernel(
        dtype *dst, int dst_height, int dst_width, size_t *dst_strides,  
        dtype *src, int src_height, int src_width, size_t *src_strides,  
        int n_batches, int n_channels, int n_strides, const float *trans,
        dtype fill_value = 127
    )
    {
        // we just make a indices in [N, C, H, W] format
        int indices[4] = {
            (int)blockIdx.z / n_channels,  // n index 
            (int)blockIdx.z % n_channels,  // c index
            (int)(threadIdx.y + blockDim.y * blockIdx.y),  // h index 
            (int)(threadIdx.x + blockDim.x * blockIdx.x)  // w index
        };
        // indices out of range.
        if (
            indices[0] >= n_batches ||
            indices[1] >= n_channels || 
            indices[2] >= dst_height || 
            indices[3] >= dst_width
        )
            return;
        // get the offset of dst.
        size_t dst_idx = 0;
        for (int i = -1; i >= -1 * n_strides; --i)
            dst_idx += indices[i + 4] * dst_strides[i + n_strides];  // implicit to size_t

        // map the 2-D coordinates of src.
        float x_src_idx, y_src_idx;
        affine_poject(trans, indices[3], indices[2], &x_src_idx, &y_src_idx);
        // src_idx out of range.
        if (x_src_idx >= src_width || x_src_idx < 0 || y_src_idx >= src_height || y_src_idx < 0)
        {
            // *((dtype *)dst + dst_idx) = fill_value;
            *(dst + dst_idx) = fill_value;
            return;
        }
        // (int)x_src_idx in [0, src_width - 1], (int)y_src_idx in [0, src_height - 1] 
        float x_left = floorf(x_src_idx);
        float x_right = limit(floorf(x_src_idx) + 1, (float)0.0, (float)src_width - 1);
        float y_top = floorf(y_src_idx);
        float y_bot = limit(floorf(y_src_idx) + 1, (float)0.0, (float)src_height - 1);
        
        float w_left = x_src_idx - x_left;
        float w_right = x_right - x_src_idx;
        float w_top = y_src_idx - y_top;
        float w_bot = y_bot - y_src_idx;

        indices[3] = (int)x_left;  
        indices[2] = (int)y_top; 
        size_t src_idx = 0;
        for (int i = -1; i >= -1 * n_strides; --i)
            src_idx += indices[i + 4] * src_strides[i + n_strides]; // implicit to size_t
        // float pixel_left_top = *((dtype *)src + src_idx);
        float pixel_left_top = *(src + src_idx);

        indices[3] = (int)x_right;  
        indices[2] = (int)y_top; 
        src_idx = 0;
        for (int i = -1; i >= -1 * n_strides; --i)
            src_idx += indices[i + 4] * src_strides[i + n_strides]; // implicit to size_t
        // float pixel_right_top = *((dtype *)src + src_idx);
        float pixel_right_top = *(src + src_idx);

        indices[3] = (int)x_left;  
        indices[2] = (int)y_bot; 
        src_idx = 0;
        for (int i = -1; i >= -1 * n_strides; --i)
            src_idx += indices[i + 4] * src_strides[i + n_strides]; // implicit to size_t
        // float pixel_left_bot = *((dtype *)src + src_idx);
        float pixel_left_bot = *(src + src_idx);

        indices[3] = (int)x_right;  
        indices[2] = (int)y_bot; 
        src_idx = 0;
        for (int i = -1; i >= -1 * n_strides; --i)
            src_idx += indices[i + 4] * src_strides[i + n_strides]; // implicit to size_t
        // float pixel_right_bot = *((dtype *)src + src_idx);
        float pixel_right_bot = *(src + src_idx);

        float pixel = pixel_left_top * w_right * w_bot + pixel_right_top * w_left * w_bot 
                    + pixel_left_bot * w_right * w_top + pixel_right_bot * w_left * w_top;

        // *((dtype *)dst + dst_idx) = (dtype)limit(pixel, (float)0.0, (float)255.0);
        *(dst + dst_idx) = (dtype)limit(pixel, (float)0.0, (float)255.0);
    }

    static void assert_interpolation_2d(        
        MixMat &src, const std::vector<int> &dims, 
        float scale_factor_x, float scale_factor_y,
        int inter_type
    )
    {
        assert(dims.size() == 2);
        if (dims[0] <= 0 || dims[1] <= 0)
        {
            assert(scale_factor_x > 0 && scale_factor_y > 0);
        }
        assert(inter_type == InterType::BILINEAR || inter_type == InterType::NEAREST);
        assert(!src.empty());
    }

    // 4-D limit, unable to handle higher than 4-D
    __host__ MixMat interpolate_2d(
        MixMat &src, const std::vector<int> &dims, 
        float scale_factor_x, float scale_factor_y,
        int inter_type
    )
    {
        assert_interpolation_2d(src, dims, scale_factor_x, scale_factor_y, inter_type);
        int device_id = src.device_id();
        cuda_tools::AutoExchangeDevice ex(device_id);
        std::vector<int> src_dims = src.dims();
        std::vector<size_t> src_strides = src.strides();
        int n_strides = src_strides.size();
        int n_src_dims = src.n_dims();
        int src_width = src_dims[-1 + n_src_dims];
        int src_height = src_dims[-2 + n_src_dims];
        int n_channels = n_src_dims >= 3 ? src_dims[-3 + n_src_dims] : 1;
        int n_batches = n_src_dims > 3 ? src_dims[-4 + n_src_dims] : 1; 

        int dst_width = dims[1] == 0 ? src_width * scale_factor_x : dims[1];
        int dst_height = dims[0] == 0 ? src_height * scale_factor_y : dims[0];
        std::vector<int> dst_dims = src_dims;
        dst_dims[-1 + n_src_dims] = dst_width;
        dst_dims[-2 + n_src_dims] = dst_height;
        tinycv::DataType dtype = src.dtype();

        MixMat dst(dst_dims, dtype, nullptr, device_id); 
        std::vector<size_t> dst_strides = dst.strides();
        float trans_array[6] = {0};
        trans_array[0] = scale_factor_x > 0 ? scale_factor_x : (float)dst_width / src_width;
        trans_array[4] = scale_factor_y > 0 ? scale_factor_y : (float)dst_height / src_height;
        trans_array[5] = 0;
    
        float inv_trans[6];
        inverse_affine_trans(trans_array, inv_trans);
        float *p_trans = nullptr; 
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_trans, sizeof(inv_trans)));
        CHECK_CUDA_RUNTIME(cudaMemcpy(p_trans, inv_trans, sizeof(inv_trans), cudaMemcpyHostToDevice));

        size_t *p_dst_strides = nullptr;
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_dst_strides, sizeof(size_t) * n_strides));
        CHECK_CUDA_RUNTIME(cudaMemcpy(p_dst_strides, dst_strides.data(), sizeof(size_t) * n_strides, cudaMemcpyHostToDevice));
        size_t *p_src_strides = nullptr;
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_src_strides, sizeof(size_t) * n_strides));
        CHECK_CUDA_RUNTIME(cudaMemcpy(p_src_strides, src_strides.data(), sizeof(size_t) * n_strides, cudaMemcpyHostToDevice));

        dim3 block_dim(32, 32);
        dim3 grid_dim((dst_width + 31) / 32, (dst_height + 31) / 32, n_channels * n_batches);
        if (inter_type == InterType::NEAREST)
        {
            switch (dst.dtype())
            {
            case tinycv::DataType::FLOAT32:
                CHECK_CUDA_KERNEL(
                nearest_interpolate_2d_kernel<<<grid_dim, block_dim>>>(
                    dst.device<float>(), dst_height, dst_width, p_dst_strides,
                    src.device<float>(), src_height, src_width, p_src_strides,
                    n_batches, n_channels, n_strides, p_trans
                ));
                break;
            case tinycv::DataType::UINT8:
                CHECK_CUDA_KERNEL(
                nearest_interpolate_2d_kernel<<<grid_dim, block_dim>>>(
                    dst.device<uint8_t>(), dst_height, dst_width, p_dst_strides,
                    src.device<uint8_t>(), src_height, src_width, p_src_strides,
                    n_batches, n_channels, n_strides, p_trans
                ));
            default:
                break;
            }
        }
        else if (inter_type == InterType::BILINEAR)
        {
            switch(dst.dtype())
            {
                case tinycv::DataType::FLOAT32:
                    CHECK_CUDA_KERNEL(
                    bilinear_interpolate_2d_kernel<<<grid_dim, block_dim>>>(
                        dst.device<float>(), dst_height, dst_width, p_dst_strides,
                        src.device<float>(), src_height, src_width, p_src_strides,
                        n_batches, n_channels, n_strides, p_trans
                    ));
                    break;
                case tinycv::DataType::UINT8:
                    CHECK_CUDA_KERNEL(
                    bilinear_interpolate_2d_kernel<<<grid_dim, block_dim>>>(
                        dst.device<uint8_t>(), dst_height, dst_width, p_dst_strides,
                        src.device<uint8_t>(), src_height, src_width, p_src_strides,
                        n_batches, n_channels, n_strides, p_trans
                    ));
                    break;
                default:
                    break;
            }
        }
        else
            INFO_FATAL("Invalid arg inter_type");

        CHECK_CUDA_RUNTIME(cudaFree(p_trans));      
        CHECK_CUDA_RUNTIME(cudaFree(p_dst_strides));     
        CHECK_CUDA_RUNTIME(cudaFree(p_src_strides));     

        return dst;
    }

    __host__ MixMat warp_affine(
        MixMat &src, 
        const std::vector<int> &dims, 
        const std::vector<float> &trans,
        int inter_type,
        float fill_value
    )
    {
        assert(dims.size() == 2 && !src.empty() && src.n_dims() <= 4);
        assert(trans.size() == 6);
        int device_id = src.device_id();
        cuda_tools::AutoExchangeDevice ex(device_id);
        std::vector<int> src_dims = src.dims();
        std::vector<size_t> src_strides = src.strides();
        int n_strides = src_strides.size();
        int n_src_dims = src.n_dims();
        int src_width = src_dims[-1 + n_src_dims];
        int src_height = src_dims[-2 + n_src_dims];
        int n_channels = n_src_dims >= 3 ? src_dims[-3 + n_src_dims] : 1;
        int n_batches = n_src_dims > 3 ? src_dims[-4 + n_src_dims] : 1; 

        int dst_width = dims[1];
        int dst_height = dims[0];
        std::vector<int> dst_dims = src_dims;
        dst_dims[-1 + n_src_dims] = dst_width;
        dst_dims[-2 + n_src_dims] = dst_height;
        tinycv::DataType dtype = src.dtype();

        MixMat dst(dst_dims, dtype, nullptr, device_id); 
        std::vector<size_t> dst_strides = dst.strides();
        float trans_array[6] = {0};
        for (int i = 0; i < 6; ++i)
            trans_array[i] = trans[i];
    
        float inv_trans[6];
        inverse_affine_trans(trans_array, inv_trans);
        float *p_trans = nullptr; 
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_trans, sizeof(inv_trans)));
        CHECK_CUDA_RUNTIME(cudaMemcpy(p_trans, inv_trans, sizeof(inv_trans), cudaMemcpyHostToDevice));

        size_t *p_dst_strides = nullptr;
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_dst_strides, sizeof(size_t) * n_strides));
        CHECK_CUDA_RUNTIME(cudaMemcpy(p_dst_strides, dst_strides.data(), sizeof(size_t) * n_strides, cudaMemcpyHostToDevice));
        size_t *p_src_strides = nullptr;
        CHECK_CUDA_RUNTIME(cudaMalloc(&p_src_strides, sizeof(size_t) * n_strides));
        CHECK_CUDA_RUNTIME(cudaMemcpy(p_src_strides, src_strides.data(), sizeof(size_t) * n_strides, cudaMemcpyHostToDevice));

        dim3 block_dim(32, 32);
        dim3 grid_dim((dst_width + 31) / 32, (dst_height + 31) / 32, n_channels * n_batches);
        if (inter_type == InterType::NEAREST)
        {
            switch(dst.dtype())
            {
                case tinycv::DataType::UINT8:
                    CHECK_CUDA_KERNEL(
                    nearest_interpolate_2d_kernel<uint8_t><<<grid_dim, block_dim>>>(
                        dst.device<uint8_t>(), dst_height, dst_width, p_dst_strides,
                        src.device<uint8_t>(), src_height, src_width, p_src_strides,
                        n_batches, n_channels, n_strides, p_trans, fill_value
                    ));
                    break;
                
                case tinycv::DataType::FLOAT32:
                    CHECK_CUDA_KERNEL(
                    nearest_interpolate_2d_kernel<float><<<grid_dim, block_dim>>>(
                        dst.device<float>(), dst_height, dst_width, p_dst_strides,
                        src.device<float>(), src_height, src_width, p_src_strides,
                        n_batches, n_channels, n_strides, p_trans, fill_value
                    ));
                    break;

                default:
                    break;
            }

        }
        else if (inter_type == InterType::BILINEAR)
        {
            switch(dst.dtype())
            {
                case tinycv::DataType::UINT8:
                    CHECK_CUDA_KERNEL(
                    bilinear_interpolate_2d_kernel<uint8_t><<<grid_dim, block_dim>>>(
                        dst.device<uint8_t>(), dst_height, dst_width, p_dst_strides,
                        src.device<uint8_t>(), src_height, src_width, p_src_strides,
                        n_batches, n_channels, n_strides, p_trans, fill_value
                    ));
                    break;
                case tinycv::DataType::FLOAT32:
                    CHECK_CUDA_KERNEL(
                    bilinear_interpolate_2d_kernel<float><<<grid_dim, block_dim>>>(
                        dst.device<float>(), dst_height, dst_width, p_dst_strides,
                        src.device<float>(), src_height, src_width, p_src_strides,
                        n_batches, n_channels, n_strides, p_trans, fill_value
                    ));
                    break;
                    
                default:
                    break;
            }
        }
        else
            INFO_FATAL("Invalid arg inter_type");

        CHECK_CUDA_RUNTIME(cudaFree(p_trans));      
        CHECK_CUDA_RUNTIME(cudaFree(p_dst_strides));     
        CHECK_CUDA_RUNTIME(cudaFree(p_src_strides));     

        return dst;
    }

    __host__ MixMat warp_affine_to_center_align(
        MixMat &src, 
        const std::vector<int> &dims, 
        int inter_type,
        float fill_value
    )
    {
        const std::vector<int> src_dims = src.dims();
        int n_src_dims = src.n_dims();
        int src_height = src_dims[-2 + n_src_dims];
        int src_width = src_dims[-1 + n_src_dims];
        int n_dst_dims = dims.size();
        int dst_height = dims[-2 + n_dst_dims];
        int dst_width = dims[-1 + n_dst_dims];
        float scale_factor_x = (float)dst_width / src_width;
        float scale_factor_y = (float)dst_height / src_height;
        float scale_factor = scale_factor_x < scale_factor_y ? scale_factor_x : scale_factor_y;
        float offset_x = dst_width / 2 - src_width * scale_factor / 2;
        float offset_y = dst_height / 2 - src_height * scale_factor / 2; 
        std::vector<float> trans = {scale_factor, 0, offset_x, 
                                    0, scale_factor, offset_y}; 

        return warp_affine(src, dims, trans, inter_type, fill_value);
    }
};
