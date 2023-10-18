#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP
#include <memory>
#include <string>

namespace tinycv
{
    namespace trt
    {
        template <typename type>
        std::shared_ptr<type> make_nvshared(type *ptr)
        {
            return std::shared_ptr<type> (ptr, [](type *ptr) -> void {ptr->destroy();});
        } 

        enum Mode_t
        {
            TRT_FP16 = 0,
            TRT_FP32,
            TRT_INT8,
        };

        char *mode_string(Mode_t m_type);

        bool compile(
            Mode_t mode,
            int min_batch_size,
            int opt_batch_size,
            int max_batch_size,
            const std::string &source, 
            const std::string &save_to,
            const size_t max_workspace_size 
        );
    } 
}



#endif  //TRT_BUILDER_HPP