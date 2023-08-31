//
// Created by AC on 2023/8/30.
//

#include "picodet_api.h"


void init_model(model_handle_t*model_handle, int input_width,
                int input_height, float score_threshold, float nms_threshold) {
    *model_handle = new PicoDet(input_width,
                                input_height,
                                score_threshold,
                                nms_threshold);
}

int image_predict_file(model_handle_t model_handle, const char *image_path,
                       void *out_buffer, const char *save_file_name) {

    if (!Utils::has_image_extension(image_path)) {
        return -1;
    }
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    std::vector<BoxInfo> results;
    static_cast<PicoDet *>(model_handle)->detect(img, results);
    cv::Mat image = PicoDet::draw_bboxes(img, results);
    if (save_file_name && Utils::has_image_extension(save_file_name)) {
        cv::imwrite(save_file_name, image);
        std::cout << "file saved in" << save_file_name << std::endl;
    }
    memcpy(out_buffer, image.data, image.total() * image.elemSize());
    int person_num = 0;
    for (auto &result: results) {
        if (result.label == 0) {
            person_num++;
        }
    }
    return person_num;
}

int image_predict_buffer(model_handle_t model_handle, void *in_buffer, void *out_buffer, int w, int h,
                         const char *save_file_name) {
    cv::Mat img = cv::Mat(h, w, CV_8UC3, in_buffer);
    std::vector<BoxInfo> results;
    static_cast<PicoDet *>(model_handle)->detect(img, results);
    cv::Mat image = PicoDet::draw_bboxes(img, results);
    if (save_file_name && Utils::has_image_extension(save_file_name)) {
        cv::imwrite(save_file_name, image);
        std::cout << "file saved in" << save_file_name << std::endl;
    }
    memcpy(out_buffer, image.data, image.total() * image.elemSize());
    int person_num = 0;
    for (auto &result: results) {
        if (result.label == 0) {
            person_num++;
        }
    }
    return person_num;
}

void free_model(void *model_handle) {
    delete static_cast<PicoDet *>(model_handle);
}


