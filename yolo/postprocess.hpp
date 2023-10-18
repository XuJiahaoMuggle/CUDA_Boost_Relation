#include <tuple>
#include <cstdint>
namespace tinycv
{
    namespace yolo
    {
        std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);

        std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);

        struct Box
        {
            Box() = default;
            Box(float l, float t, float r, float b, float conf, int c_label)
                : left(l), top(t), right(r), bottom(b), confidence(conf), class_label(c_label) {}   
         
            float left = 0, top = 0, right = 0, bottom = 0, confidence = 0;
            int class_label = -1;
        };

        void decode_bbox(
            float *bbox_predict,
            int n_bboxes,  // the number of boxes.
            int n_classes,  // the number of classes
            int n_elemnt_bbox,  // the element number of each box. 
            float *d2i,
            float *bbox_out,
            int n_max_bbox,
            float conf_threshold = 0.5,
            void *stream = nullptr
        );

        void nms(
            float *ptr_bbox, 
            int n_max_bboxes,
            float nms_threshold = 0.5,
            void *stream = nullptr
        );

        void postprocess(
            float *bbox_predict,
            int n_bboxes,
            int n_classes,
            int n_bbox_info,
            float *d2i,
            float *bbox_out,
            int n_max_bboxes,
            float conf_threshold,
            float nms_threshold,
            void *stream,
            bool sync = false
        );
    }
}