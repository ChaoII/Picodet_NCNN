//
// Created by AC on 2023/8/30.
//

#include "picodet.h"
#include "picodet_api.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>




int image_demo(PicoDet &detector, const char *image_path,
               int has_postprocess = 0) {
    std::vector<cv::String> filenames;
    cv::glob(image_path, filenames, false);
    bool is_postprocess = has_postprocess > 0;
    for (auto &img_name: filenames) {
        cv::Mat image = cv::imread(img_name, cv::IMREAD_COLOR);
        std::vector<BoxInfo> results;
        detector.detect(image, results, is_postprocess);
        std::string save_path = img_name;
        draw_bboxes(image, results, "result.jpg");
    }
    return 0;
}




int main() {
    PicoDet detector =

    const char *images = "../R.jpg";
    image_demo(detector, images, false);
}