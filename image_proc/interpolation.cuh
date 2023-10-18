#ifndef INTERPOLATION_CUH
#define INTERPOLATION_CUH
// Welcome to the world of CUDA

#include <vector>
#include <mix_memory/mix_mat.hpp>

namespace tinycv
{
    enum InterType
    {
        NEAREST,
        BILINEAR
    };
    
    __host__ MixMat interpolate_2d(
        MixMat &src, const std::vector<int> &dims,
        float scale_factor_x = -1, 
        float scale_factor_y = -1, 
        int inter_type = InterType::BILINEAR
    );

    __host__ MixMat warp_affine(
        MixMat &src, 
        const std::vector<int> &dims, 
        const std::vector<float> &trans,
        int inter_type = InterType::BILINEAR,
        float fill_value = 127
    );

    __host__ MixMat warp_affine_to_center_align(
        MixMat &src, 
        const std::vector<int> &dims, 
        int inter_type = InterType::BILINEAR,
        float fill_value = 127
    );

};

#endif  // INTERPOLATION_CUH