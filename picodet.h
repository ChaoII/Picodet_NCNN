//
// Created by AC on 2023/8/30.
//
#pragma once

#include <net.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "picodet_id.h"
#include "picodet_mem.h"

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
    PicoDet(int input_width, int input_height, float score_threshold_, float nms_threshold_);

    ~PicoDet();

    ncnn::Net *Net;

    int detect(cv::Mat image, std::vector<BoxInfo> &result_list);

    static void draw_bboxes(const cv::Mat &im, const std::vector<BoxInfo> &bboxes,
                            const std::string &save_path = "None");

    static std::vector<int> GenerateColorMap(int num_class_);

private:
    void preprocess(cv::Mat &image, ncnn::Mat &in);

    void decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride,
                      float threshold,
                      std::vector<std::vector<BoxInfo>> &results);

    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x,
                         int y, int stride);

    static void nms(std::vector<BoxInfo> &result, float nms_threshold);

    void nms_boxes(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred,
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

    std::vector<std::array<int, 3>> non_postprocess_heads_info{
            {__models_picodet_param_id::BLOB_transpose_0_tmp_0, __models_picodet_param_id::BLOB_transpose_1_tmp_0, 8},
            {__models_picodet_param_id::BLOB_transpose_2_tmp_0, __models_picodet_param_id::BLOB_transpose_3_tmp_0, 16},
            {__models_picodet_param_id::BLOB_transpose_4_tmp_0, __models_picodet_param_id::BLOB_transpose_5_tmp_0, 32},
            {__models_picodet_param_id::BLOB_transpose_6_tmp_0, __models_picodet_param_id::BLOB_transpose_7_tmp_0, 64},
    };
};
