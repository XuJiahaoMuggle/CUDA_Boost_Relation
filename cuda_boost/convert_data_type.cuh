#ifndef CONVERT_DATA_TYPE_CUH
#define CONVERT_DATA_TYPE_CUH
#include <mix_memory/mix_mat.hpp>

/**
 * @brief Convert data type from dst_dtype to src_dtype.
 * Support float32 -> uint8, int8, int32, float16(__half)
 * uint8 -> float32, int32, int8
 * int32 -> float32, float16, uint8, int8
 * float16(__half) -> float32, int32
 * @param new_data (void *): pointer on Device.
 * @param dst_dtype (tinycv::DataType)
 * @param old_data (void *): pointer on Device.
 * @param src_dtype (tinycv::DataType)
 * @param size (size_t): the number of elements to be converted.
 * @param stream (cudaStream_t): use stream or not.
 */
__host__ void convert_data_type_async(
    void *new_data, tinycv::DataType dst_dtype, 
    void *old_data, tinycv::DataType src_dtype, 
    size_t size, cudaStream_t stream = nullptr, 
    bool sync = true
);

#endif  // CONVERT_DATA_TYPE_CUH
