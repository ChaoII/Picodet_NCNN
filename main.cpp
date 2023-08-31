//
// Created by AC on 2023/8/30.
//


#include "picodet_api.h"
#include <iostream>

void test_image_predict_file() {
    std::cout << "==============test for 'test_image_predict_file'==============" << std::endl;
    std::cout << "init model handle" << std::endl;
    model_handle_t model_handle = nullptr;
    init_model(&model_handle, 320, 320, 0.5f, 0.45f);
    cv::Mat img = cv::imread("R.jpg");
    void *out_buffer = malloc(img.total() * img.elemSize());
    int count = image_predict_file(model_handle, "R.jpg", out_buffer, nullptr);
    std::cout << "person num: " << count << std::endl;
    cv::Mat dst_img = cv::Mat(img.rows, img.cols, CV_8UC3, out_buffer);
    cv::imwrite("result1.jpg", dst_img);
    std::cout << "predict image file is saved in 'result1.jpg' as the binary directory" << std::endl;
    free_model(model_handle);
    std::cout << "free model handle" << std::endl;
}

void test_image_predict_buffer() {
    std::cout << "==============test for 'test_image_predict_file'==============" << std::endl;
    std::cout << "init model handle" << std::endl;
    model_handle_t model_handle = nullptr;
    init_model(&model_handle, 320, 320, 0.5f, 0.45f);
    cv::Mat img = cv::imread("R.jpg");
    void *out_buffer = malloc(img.total() * img.elemSize());
    int count = image_predict_buffer(model_handle, img.data, out_buffer, img.cols, img.rows, nullptr);
    std::cout << "person num: " << count << std::endl;
    cv::Mat dst_img = cv::Mat(img.rows, img.cols, CV_8UC3, out_buffer);
    cv::imwrite("result2.jpg", dst_img);
    std::cout << "predict image file is saved in 'result2.jpg' as the binary directory" << std::endl;
    free_model(model_handle);
    std::cout << "free model handle" << std::endl;
}


int main() {
    test_image_predict_file();
    test_image_predict_buffer();
}