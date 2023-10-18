#include "trt_builder.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <utils/utils.h>

class TRTLogger: public nvinfer1::ILogger
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

namespace tinycv
{
    namespace trt
    {
        char *mode_string(Mode_t m_type)
        {
            switch (m_type)
            {
            case Mode_t::TRT_FP32:
                return "TRT_FP32";
            case Mode_t::TRT_FP16:
                return "TRT_FP16";
            default:
                return "Unknown TRT Mode type";
            }
        }   

        bool compile(
            Mode_t mode,
            int min_batch_size,
            int opt_batch_size,
            int max_batch_size,
            const std::string &source, 
            const std::string &save_to,
            const size_t max_workspace_size    
        )
        {
            TRTLogger trt_logger;
            if(!initLibNvInferPlugins(static_cast<void *>(&trt_logger), ""))
            {
                INFO_ERROR("Init plugin failed.");
                return false;
            }
            INFO("Compiling onnx file %s with %s mode.", source.c_str(), mode_string(mode));
            // Create builder
            std::shared_ptr<nvinfer1::IBuilder> builder = make_nvshared(nvinfer1::createInferBuilder(trt_logger));
            if (!builder)
            {
                INFO_ERROR("Can't create trt builder.");
                return false;
            }
            // Create network with explicit and dynamic batch.
            std::shared_ptr<nvinfer1::INetworkDefinition> network = make_nvshared(
                builder->createNetworkV2(1U << static_cast<int32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
            );
            // Create optimization profile.
            nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
            // Create builder config
            std::shared_ptr<nvinfer1::IBuilderConfig> builder_config = make_nvshared(builder->createBuilderConfig());
            if (mode = Mode_t::TRT_FP16)
            {
                if (!builder->platformHasFastFp16())
                    INFO_WARNING("Unsupport mode type FP16.");
                else    
                    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            // Cretae onnx parser.
            std::shared_ptr<nvonnxparser::IParser> onnx_parser = make_nvshared(nvonnxparser::createParser(*network, trt_logger));
            if (!onnx_parser)
            {
                INFO_ERROR("Create Onnx parser failed.");
                return false;
            }
            // Parse onnx.
            if (!onnx_parser->parseFromFile(source.c_str(), 1))
            {
                INFO_ERROR("Can't parse onnx file called %s.", source.c_str());
                return false;
            }
            // Set max_workspace_size
            builder_config->setMaxWorkspaceSize(max_workspace_size);
            INFO("Workspace size: %fMB", max_workspace_size / 1024.0f / 1024.0f);
            int n_inputs = network->getNbInputs();  // number of inputs.
            int n_outputs = network->getNbOutputs();  // number of outputs.
            nvinfer1::ITensor *input_tensor = network->getInput(0);
            nvinfer1::Dims input_dims = input_tensor->getDimensions();
            
            // Config profile.
            input_dims.d[0] = min_batch_size;  // min batch size
            profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims); 
            INFO("Set min batch size: %d", input_dims.d[0]);
            input_dims.d[0] = opt_batch_size;  // opt batch size
            profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);  
            INFO("Set opt batch size: %d", input_dims.d[0]);
            input_dims.d[0] = max_batch_size;  // max batch size
            profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);  
            INFO("Set max batch size: %d", input_dims.d[0]);

            // Add optimization profile to builde config.
            builder_config->addOptimizationProfile(profile);
            INFO("Builder engine, it will take some time.");
            std::shared_ptr<nvinfer1::ICudaEngine> engine = make_nvshared(builder->buildEngineWithConfig(*network, *builder_config));
            if (!engine)
            {
                INFO("Build engine failed.");
                return false;
            }
            // Serialize engine.
            std::shared_ptr<nvinfer1::IHostMemory> host_mem = make_nvshared(engine->serialize());
            FILE *f = fopen(save_to.c_str(), "wb");
            if (!f)
            {
                INFO_ERROR("Open file failed.");
                return false;
            }
            // Save file. 
            if (host_mem->data())
            {
                if (fwrite(host_mem->data(), 1, host_mem->size(), f) != host_mem->size())
                {
                    fclose(f);
                    return false;
                }
            }
            fclose(f);
            INFO("Build trt engine done.");
            return true;
        }
    }
}