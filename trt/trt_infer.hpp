#ifndef TRT_INFER_HPP
#define TRT_INFER_HPP
#include <mix_memory/mix_mat.hpp>
#include <memory>
#include <string>
#include <vector>

namespace tinycv
{
    namespace trt
    {
        class TRTInfer
        {
        public:
            virtual bool forward(std::vector<void *> bindings, void *stream = nullptr) = 0;
            virtual int get_max_batch_size() = 0;

            virtual size_t get_devide_memory_size() = 0;

            virtual std::vector<int> get_binding_dims(int idx) = 0;
            virtual bool set_bindings_dims(int idx, std::vector<int> &dims) = 0;

            virtual int get_n_bindings() = 0;
            virtual int get_n_inputs() = 0;
            virtual int get_n_outputs() = 0;
        
            virtual bool is_dynamic() = 0;
            
            virtual void print() = 0;
            virtual int device_id() = 0;
        };

        bool init_nv_plugin();

        std::shared_ptr<TRTInfer> load_infer_from_file(const std::string &file, int device_id = 0);
        std::shared_ptr<TRTInfer> load_infer_from_memory(void *data, size_t size, int device_id = 0);
    }
}



#endif  // TRT_INFER_HPP