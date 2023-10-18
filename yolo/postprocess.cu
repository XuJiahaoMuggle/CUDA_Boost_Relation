#include "postprocess.hpp"
#include <utils/utils.h>
#include <cstdint>
#include <cstdio>

#define BLOCK_SIZE 512
const int N_BBOX_ELEMENT = 8;  // left top right bottom label confidence keep_flag reserve totallt 32B

static __host__ __device__ void affine_project(float *affine_mat, float x, float y, float *proj_x, float *proj_y)
{
    *proj_x = affine_mat[0] * x + affine_mat[1] * y + affine_mat[2];
    *proj_y = affine_mat[3] * x + affine_mat[4] * y + affine_mat[5]; 
}

/**
 * @brief Decode output from network.
 * @param bbox_predict (float *): the output from network, shape: [N, 25200, 5 + number of classes]
 * @param n_bboxes (int): the number of boxes to decode. 
 * @param n_classes (int): the total number of classes.
 * @param n_bbox_info (int): the length of each box.
 * @param d2i (float *)
 * @param bbox_out (float *): the decoded result.
 * @param n_max_bboxes (int): the maximum number of boxes.
 * @param conf_threshold (float)
 */
static __global__ void decode_bbox_kernel(
    float *bbox_predict,
    int n_bboxes,
    int n_classes,
    int n_bbox_info,
    float *d2i,
    float *bbox_out,
    int n_max_bboxes,
    float conf_threshold
){
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_bboxes)
        return;
    float *ptr_bbox = bbox_predict + n_bbox_info * idx;
    float conf_objectness = ptr_bbox[4];
    if (conf_objectness < conf_threshold)
        return;
    float *ptr_conf_class = &ptr_bbox[5];
    float conf_class = *ptr_conf_class;
    int label = 0;
    for (int i = 1; i < n_classes; ++i)
    {
        if (*++ptr_conf_class > conf_class)
        {
            label = i;
            conf_class = *ptr_conf_class;
        }
    }
    
    float conf = conf_class * conf_objectness;
    if (conf < conf_threshold)
        return;
    
    // atomic operation required here.
    int n_choosed = (int)atomicAdd(bbox_out, 1);
    if (n_choosed >= n_max_bboxes)
        return;
    // (c_x, c_y, w, h)
    float c_x = ptr_bbox[0];
    float c_y = ptr_bbox[1];
    float w = ptr_bbox[2];
    float h = ptr_bbox[3];
    // (top, left) and (bottom, right) can specify a bbox.
    float left = c_x - 0.5 * w;
    float top = c_y - 0.5 * h;
    float right = c_x + 0.5 * w;
    float bottom = c_y + 0.5 *h;

    // project to source image
    affine_project(d2i, left, top, &left, &top);
    affine_project(d2i, right, bottom, &right, &bottom);

    // printf("%f %f %f %f\n", left, top, right, bottom);
    float *ptr_out = (float *)((uint8_t *)bbox_out + 32) + n_choosed * N_BBOX_ELEMENT;
    ptr_out[0] = left;
    ptr_out[1] = top;
    ptr_out[2] = right;
    ptr_out[3] = bottom;
    ptr_out[4] = conf;
    ptr_out[5] = label;  
    ptr_out[6] = 1;  // keep flag: 1 on keep, 0 on discard.  
}

static __device__ __host__ float bbox_iou(float left1, float top1, float right1, float bottom1, float left2, float top2, float right2, float bottom2)
{
    float inter_left = max(left1, left2);
    float inter_top = max(top1, top2);
    float inter_right = min(right1, right2);
    float inter_bottom = min(bottom1, bottom2);
    float inter_area = max(inter_bottom - inter_top, 0.0f) * max(inter_right - inter_left, 0.0f);
    if (inter_area <= 0.0f)
        return 0.0f;
    float union_area = (right1 - left1) * (bottom1 - top1) + (right2 - left2) * (bottom2 - top2) - inter_area;

    return inter_area / union_area; 
}

static __global__ void nms_kernel(
    float *ptr_bbox, 
    int n_max_bboxes,
    float nms_threshold
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n_bboxes = min(*ptr_bbox, (float)n_max_bboxes);
    if (idx >= n_bboxes)
        return;
    float *p_cur = (float*)((uint8_t *)(ptr_bbox) + 32) + idx * N_BBOX_ELEMENT;
    for (int i = 0; i < n_bboxes; ++i)
    {
        float *p_item = (float*)((uint8_t *)(ptr_bbox) + 32) + i * N_BBOX_ELEMENT;
        if (i == idx || p_item[5] != p_cur[5])
            continue;
        
        if (p_item[4] >= p_cur[4])
        {
            if (p_item[4] == p_cur[4] && i < idx)
                continue;
            float iou = bbox_iou(p_cur[0], p_cur[1], p_cur[2], p_cur[3], p_item[0], p_item[1], p_item[2], p_item[3]);
            if (iou > nms_threshold)
            {
                p_cur[6] = 0;
                return;
            }
        }
    }   
}

namespace tinycv
{   
    namespace yolo
    {
        std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) 
        {
            const int h_i = static_cast<int>(h * 6);
            const float f = h * 6 - h_i;
            const float p = v * (1 - s);
            const float q = v * (1 - f * s);
            const float t = v * (1 - (1 - f) * s);
            float r, g, b;
            switch (h_i) 
            {
                case 0:
                r = v, g = t, b = p;
                break;
                case 1:
                r = q, g = v, b = p;
                break;
                case 2:
                r = p, g = v, b = t;
                break;
                case 3:
                r = p, g = q, b = v;
                break;
                case 4:
                r = t, g = p, b = v;
                break;
                case 5:
                r = v, g = p, b = q;
                break;
                default:
                r = 1, g = 1, b = 1;
                break;
            }
            return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                                static_cast<uint8_t>(r * 255));
        }

        std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) 
        {
            float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
            float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
            return hsv2bgr(h_plane, s_plane, 1);
        }

        void decode_bbox(
            float *bbox_predict,
            int n_bboxes,  // the number of boxes.
            int n_classes,  // the number of classes
            int n_bbox_info,  // the element number of each box. 
            float *d2i,
            float *bbox_out,
            int n_max_bboxes,
            float conf_threshold,
            void *stream
        ){
            dim3 block_dim = BLOCK_SIZE;
            dim3 grid_dim = (n_bboxes - 1 + BLOCK_SIZE) / BLOCK_SIZE;
            cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
            CHECK_CUDA_KERNEL(
                decode_bbox_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    bbox_predict, n_bboxes, n_classes, n_bbox_info,
                    d2i, bbox_out, n_max_bboxes, conf_threshold
                )
            );
        }

        void nms(
            float *ptr_bbox, 
            int n_max_bboxes,
            float nms_threshold,
            void *stream
        ){
            dim3 block_dim = BLOCK_SIZE;
            dim3 grid_dim = (n_max_bboxes - 1 + BLOCK_SIZE) / BLOCK_SIZE;
            cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(stream);
            CHECK_CUDA_KERNEL(
                nms_kernel<<<grid_dim, block_dim, 0, stream_>>>(
                    ptr_bbox,
                    n_max_bboxes,
                    nms_threshold
                )
            );
        }

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
            bool sync
        ){
            decode_bbox(bbox_predict, n_bboxes, n_classes, n_bbox_info, d2i, bbox_out, n_max_bboxes, conf_threshold, stream);
            nms(bbox_out, n_max_bboxes, nms_threshold, stream);
            if (sync)
                CHECK_CUDA_RUNTIME(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
        }
    };
}

