#ifndef YOLO_INFER_HPP
#define YOLO_INFER_HPP
#include "preprocess.hpp"
#include "postprocess.hpp"
#include <vector>
#include <memory>
#include <string>
#include <mix_memory/mix_mat.hpp>
namespace tinycv
{
    namespace yolo
    {
        using BoxArray = std::vector<Box>;
        class YoloInfer
        {
        public:
            virtual int get_height() = 0;
            virtual int get_width() = 0;
            virtual int get_n_channels() = 0;
            virtual int get_n_classes() = 0;

            virtual BoxArray forward(Frame &frame, void *stream = nullptr) = 0;
            virtual std::vector<BoxArray> forward(std::vector<Frame> &frames, void *stream = nullptr) = 0;

            virtual BoxArray forward(MixMat &mix_mat, void *stream = nullptr) = 0;
            virtual std::vector<BoxArray> forward(std::vector<MixMat> &mix_mats, void *stream = nullptr) = 0;
        };
        
        std::shared_ptr<YoloInfer> load(const std::string &engine_file, float confidence_threshold = 0.5f, float nms_threshold = 0.5f);
    }
}



#endif  // YOLO_INFER_HPP