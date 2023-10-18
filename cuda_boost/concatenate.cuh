#ifndef CONCATENATE_CUH
#define CONCATENATE_CUH
// This is a hard work.
#include <mix_memory/mix_mat.hpp>
#include <vector>

namespace tinycv
{
    MixMat concatenate(int n_mats, const MixMat *mats, int index, cudaStream_t stream = nullptr, bool sync = true);
    
    MixMat concatenate(const std::vector<MixMat> &mats, int index, cudaStream_t stream = nullptr, bool sync = true);
};






#endif  // CONCATENATE_CUH