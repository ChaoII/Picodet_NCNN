//
// Created by AC on 2023/8/30.
//

#include "picodet_api.h"

int image_predict_file(void *model, const char *image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    std::vector<BoxInfo> results;
    static_cast<PicoDet*>(model)->detect(image, results);
    PicoDet::draw_bboxes(image, results, "result.jpg");
    return 0;
}

int init_model(void **model, int input_width, int input_height, float score_threshold, float nms_threshold) {
    *model = new PicoDet(input_width, input_height, score_threshold, nms_threshold);
}
