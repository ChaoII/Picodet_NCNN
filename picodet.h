//
// Created by AC on 2023/8/30.
//
#pragma once

#include <net.h>
#include <opencv2/core/core.hpp>

#ifdef USE_MEM

#include "picodet_id.h"
#include "picodet_mem.h"

#endif

typedef struct NonPostProcessHeadInfo {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} NonPostProcessHeadInfo;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class PicoDet {
public:
    PicoDet(const char *param, const char *bin, int input_width, int input_height, float score_threshold_,
            float nms_threshold_);

    ~PicoDet();

    ncnn::Net *Net;

    int detect(cv::Mat image, std::vector<BoxInfo> &result_list,
               bool has_postprocess);

private:
    void preprocess(cv::Mat &image, ncnn::Mat &in);

    void decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride,
                      float threshold,
                      std::vector<std::vector<BoxInfo>> &results);

    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x,
                         int y, int stride);

    static void nms(std::vector<BoxInfo> &result, float nms_threshold);

    void nms_boxes(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred,
                   float score_threshold,
                   std::vector<std::vector<BoxInfo>> &result_list);

    int image_w;
    int image_h;
    int in_w = 320;
    int in_h = 320;
    int num_class = 80;
    int reg_max = 7;

    float score_threshold;
    float nms_threshold;

    std::vector<float> bbox_output_data_;
    std::vector<float> class_output_data_;

    std::vector<std::string> nms_heads_info{"tmp_16", "concat_4.tmp_0"};
    // If not export post-process, will use non_postprocess_heads_info

#if USE_MEM
    std::vector<std::array<int, 3>> non_postprocess_heads_info{
            {__models_picodet_param_id::BLOB_transpose_0_tmp_0, __models_picodet_param_id::BLOB_transpose_1_tmp_0, 8},
            {__models_picodet_param_id::BLOB_transpose_2_tmp_0, __models_picodet_param_id::BLOB_transpose_3_tmp_0, 16},
            {__models_picodet_param_id::BLOB_transpose_4_tmp_0, __models_picodet_param_id::BLOB_transpose_5_tmp_0, 32},
            {__models_picodet_param_id::BLOB_transpose_6_tmp_0, __models_picodet_param_id::BLOB_transpose_7_tmp_0, 64},
    };
#else
    std::vector<NonPostProcessHeadInfo> non_postprocess_heads_info{
            // cls_pred|dis_pred|stride
            {"transpose_0.tmp_0", "transpose_1.tmp_0", 8},
            {"transpose_2.tmp_0", "transpose_3.tmp_0", 16},
            {"transpose_4.tmp_0", "transpose_5.tmp_0", 32},
            {"transpose_6.tmp_0", "transpose_7.tmp_0", 64},
    };
#endif
};
