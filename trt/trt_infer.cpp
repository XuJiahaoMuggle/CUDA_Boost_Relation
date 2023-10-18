#include "trt_infer.hpp"
#include "trt_builder.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <utils/utils.h>

#include <fstream>
#include <string>
#include <algorithm>
#include <unordered_map>

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
    {
        if (severity == Severity::kINTERNAL_ERROR)
            INFO_FATAL("Nvinfer INTERNAL ERROR: %s",  msg);
        else if (severity == Severity::kERROR)
            INFO_ERROR("Nvinfer ERROR: %s", msg);
        else if (severity == Severity::kWARNING)
            INFO_WARNING("Nvinfer WARNING: %s", msg);
        else if (severity == Severity::kINFO)
            INFO("Nvinfer INFO: %s", msg);
        else
            INFO_VERBOSE("Nvinfer VERBOSE: %s", msg);
    }
};

static std::vector<uint8_t> load_file(const std::string &file)
{
    // open file.
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    // move file pointer to end.
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();
    
    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char *)data.data(), length);
    }

    in.close();
    return data;
}

static tinycv::DataType convert_trt_dtype(nvinfer1::DataType dtype)
{   
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return tinycv::DataType::FLOAT32;
    case nvinfer1::DataType::kHALF:
        return tinycv::DataType::FLOAT16;
    case nvinfer1::DataType::kINT32:
        return tinycv::DataType::INT32;
    case nvinfer1::DataType::kINT8:
        return tinycv::DataType::INT8;
    default:
        INFO_ERROR("Unsupport data type.");
        return tinycv::DataType::FLOAT32;
    }
}

class TRTEngineCtx
{
public:
    ~TRTEngineCtx() { destroy_(); }

    bool build_model(void *data, size_t size)
    {
        destroy_();
        if (!data || size == 0)
            return false;
        
        TRTLogger trt_logger;
        runtime_ = make_nvshared(nvinfer1::createInferRuntime(trt_logger));
        if (!runtime_)
            return false;
        
        cuda_engine_ = make_nvshared(runtime_->deserializeCudaEngine(data, size, nullptr));
        if (!cuda_engine_)
            return false;
        
        exec_ctx_ = make_nvshared(cuda_engine_->createExecutionContext());
        return exec_ctx_ != nullptr;
    }

private:
    void destroy_()
    {
        exec_ctx_.reset();
        cuda_engine_.reset();
        runtime_.reset();
    }

    template <typename type>
    std::shared_ptr<type> make_nvshared(type *ptr)
    {
        return std::shared_ptr<type> (ptr, [] (type *ptr) -> void {ptr->destroy();});
    }

public:
    std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx_ = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine_ = nullptr;
    std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
};

namespace tinycv
{
    namespace trt
    {
        class TRTInferImpl: public TRTInfer
        {
        public:
            ~TRTInferImpl() { destroy_(); }

            bool load_from_file(const std::string &file, int device_id = 0)
            {
                std::vector<uint8_t> data = load_file(file);
                if (data.empty())
                    return false;
                trt_engine_ctx_ = std::make_shared<TRTEngineCtx>();
                if (!trt_engine_ctx_->build_model(data.data(), data.size()))
                {
                    trt_engine_ctx_.reset();  // ref_cnt - 1;
                    return false;
                }
                device_id_ = cuda_tools::get_device_id(device_id);
                parser_bindings_();
                return true;
            }

            bool load_from_memory(void *data, size_t size, int device_id = 0)
            {
                if (!data || size == 0)
                    return false;
                trt_engine_ctx_ = std::make_shared<TRTEngineCtx>();
                if (!trt_engine_ctx_->build_model(data, size))
                {
                    trt_engine_ctx_.reset();  // ref_cnt - 1;
                    return false;
                }
                device_id_ = cuda_tools::get_device_id(device_id);
                return true;
            }

            int get_max_batch_size() override
            {
                return trt_engine_ctx_->cuda_engine_->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
            }

            bool forward(std::vector<void *> bindings, void *stream) override
            {
                TRTEngineCtx *engine_ctx = trt_engine_ctx_.get();
                return engine_ctx->exec_ctx_->enqueueV2(bindings.data(), reinterpret_cast<cudaStream_t>(stream), nullptr);
            }

            size_t get_devide_memory_size() override
            {
                return trt_engine_ctx_->exec_ctx_->getEngine().getDeviceMemorySize();
            }

            int get_n_bindings() override 
            {
                return input_indices.size() + output_indices.size();
            }

            int get_n_inputs() override 
            {
                return input_indices.size();
            }

            int get_n_outputs() override
            {
                return output_indices.size();
            }

            std::vector<int> get_binding_dims(int idx) override
            {
                nvinfer1::Dims dims = trt_engine_ctx_->cuda_engine_->getBindingDimensions(idx);
                return std::vector<int>(dims.d, dims.d + dims.nbDims);
            }

            bool set_bindings_dims(int idx, std::vector<int> &dims) override
            {
                if (!is_dynamic())
                    return false;
                nvinfer1::Dims d;
                memcpy(d.d, dims.data(), sizeof(int) * dims.size());
                d.nbDims = dims.size();
                return trt_engine_ctx_->exec_ctx_->setBindingDimensions(idx, d);
            }

            bool is_dynamic() override
            {
                int n_bindings = trt_engine_ctx_->cuda_engine_->getNbBindings();
                for (int i = 0; i < n_bindings; ++i)
                {
                    nvinfer1::Dims dims = trt_engine_ctx_->cuda_engine_->getBindingDimensions(i);
                    for (int j = 0; j < dims.nbDims; ++j)
                    {
                        if (dims.d[j] == -1)
                            return true;
                    }
                }
                return false;
            }
            
            void print() override
            {
                if (!trt_engine_ctx_)
                {
                    INFO_WARNING("The engine context is nullptr.");
                    return;
                }

                INFO("Infer %p detail.", this);
                INFO("Max Batch Size: %d", get_max_batch_size());

                int n_inputs = get_n_inputs();
                INFO("Number of inputs: %d", n_inputs);

                for (int i = 0; i < n_inputs; ++i)
                {
                    const char *i_binding_name = trt_engine_ctx_->cuda_engine_->getBindingName(input_indices[i]);
                    nvinfer1::Dims dims = trt_engine_ctx_->cuda_engine_->getBindingDimensions(input_indices[i]);
                    std::string i_binding_dims = convert_dims_to_string_(dims.d, dims.nbDims);
                    INFO("Input[%d]'s information: %s, shape: %s", i, i_binding_name, i_binding_dims.c_str());
                }

                int n_outputs = get_n_outputs();
                INFO("Number of outputs: %d", n_outputs);
                for (int i = 0; i < n_outputs; ++i)
                {
                    const char *o_binding_name = trt_engine_ctx_->cuda_engine_->getBindingName(output_indices[i]);
                    nvinfer1::Dims dims = trt_engine_ctx_->cuda_engine_->getBindingDimensions(output_indices[i]);
                    std::string o_binding_dims = convert_dims_to_string_(dims.d, dims.nbDims).c_str();
                    INFO("Output[%d]'s information: %s, shape: %s", i, o_binding_name, o_binding_dims.c_str());
                }
            }

            int device_id() override
            {
                return device_id_;
            }

        private:
            void destroy_()
            {
                cuda_tools::AutoExchangeDevice ex(device_id_);
                trt_engine_ctx_.reset();
            }

            void parser_bindings_()
            {
                int n_bindings = trt_engine_ctx_->cuda_engine_->getNbBindings();
                for (int i = 0; i < n_bindings; ++i)
                {
                    if (trt_engine_ctx_->cuda_engine_->bindingIsInput(i))
                        input_indices.push_back(i);
                    else 
                        output_indices.push_back(i);
                }
            }

            std::string convert_dims_to_string_(int *dims, int n_dims) 
            {
                std::string ret;
                for (int i = 0; i < n_dims; ++i)
                {
                    ret += std::to_string(dims[i]);
                    if (i != n_dims - 1)
                        ret += " x ";
                }
                return ret;
            }

        private:
            std::vector<int> input_indices;
            std::vector<int> output_indices;
            std::shared_ptr<TRTEngineCtx> trt_engine_ctx_ = nullptr;
            int device_id_ = 0; 
        };

        bool init_nv_plugin()
        {
            TRTLogger trt_logger;
            if (!initLibNvInferPlugins(&trt_logger, ""))
            {
                INFO_ERROR("Init NV infer plugins failed.");
                return false;
            }
            return true;
        }

        bool is_endwith(const std::string &str, const std::string &target)
        {
            if (target.size() > str.size())
                return false;
            int n_str = str.size();
            int n_target = target.size();
            for (int i = 1; i < target.size(); ++i)
            {
                if (str[n_str - i] != target[n_target - i])
                    return false;
            }
            return true;
        }

        std::shared_ptr<TRTInfer> load_infer_from_file(const std::string &file, int device_id)
        {
            std::shared_ptr<TRTInferImpl> instance(new TRTInferImpl());
            if (!instance->load_from_file(file, device_id))
                instance.reset();
            return instance;
        }

        std::shared_ptr<TRTInfer> load_infer_from_memory(void *data, size_t size, int device_id)
        {
            std::shared_ptr<TRTInferImpl> instance(new TRTInferImpl());
            if (!instance->load_from_memory(data, size, device_id))
                instance.reset();
            return instance;
        }
    }
}

