//
// Created by AC on 2023/8/30.
//


#include "picodet_api.h"
#include <iostream>

void test_image_predict_file() {
    model_handle_t model_handle = nullptr;
    init_model(&model_handle, 320, 320, 0.5, 0.45);
    cv::Mat img = cv::imread("../R.jpg");
    void *out_buffer = malloc(img.total() * img.elemSize());
    int count = image_predict_file(model_handle, "../R.jpg",  out_buffer, nullptr);
    std::cout << count << std::endl;
    cv::Mat dst_img = cv::Mat(img.rows, img.cols, CV_8UC3, out_buffer);
    cv::imshow("123", dst_img);
    cv::waitKey(0);
    free_model(model_handle);
}

void test_image_predict_buffer() {
    model_handle_t model_handle = nullptr;
    init_model(&model_handle, 320, 320, 0.5, 0.45);
    cv::Mat img = cv::imread("../R.jpg");
    void *out_buffer = malloc(img.total() * img.elemSize());
    int count = image_predict_buffer(model_handle, img.data, out_buffer, img.cols, img.rows, nullptr);
    std::cout << count << std::endl;
    cv::Mat dst_img = cv::Mat(img.rows, img.cols, CV_8UC3, out_buffer);
    cv::imshow("456", dst_img);
    cv::waitKey(0);
    free_model(model_handle);
}


int main() {
    test_image_predict_file();
    test_image_predict_buffer();
}