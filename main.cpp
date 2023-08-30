//
// Created by AC on 2023/8/30.
//

#include "picodet.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


std::vector<int> GenerateColorMap(int num_class) {
    auto colormap = std::vector<int>(3 * num_class, 0);
    for (int i = 0; i < num_class; ++i) {
        int j = 0;
        int lab = i;
        while (lab) {
            colormap[i * 3] |= (((lab >> 0) & 1) << (7 - j));
            colormap[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
            colormap[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
            ++j;
            lab >>= 3;
        }
    }
    return colormap;
}

void draw_bboxes(const cv::Mat &im, const std::vector<BoxInfo> &bboxes,
                 std::string save_path = "None") {
    static const char *class_names[] = {
            "person", "bicycle", "car",
            "motorcycle", "airplane", "bus",
            "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird",
            "cat", "dog", "horse",
            "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup",
            "fork", "knife", "spoon",
            "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza",
            "donut", "cake", "chair",
            "couch", "potted plant", "bed",
            "dining table", "toilet", "tv",
            "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink",
            "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"};

    cv::Mat image = im.clone();
    int thickness = 2;
    auto colormap = GenerateColorMap(sizeof(class_names));

    for (auto &bbox: bboxes) {
        int c1 = colormap[3 * bbox.label + 0];
        int c2 = colormap[3 * bbox.label + 1];
        int c3 = colormap[3 * bbox.label + 2];
        cv::Scalar color = cv::Scalar(c1, c2, c3);
        cv::rectangle(image, cv::Rect(cv::Point((int) bbox.x1, (int) bbox.y1),
                                      cv::Point((int) bbox.x2, (int) bbox.y2)),
                      color, 1, cv::LINE_AA);
        std::string text = class_names[bbox.label] + std::to_string(bbox.score).substr(0, 4);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, thickness, &baseLine);
        int x = (int) bbox.x1;
        int y = (int) bbox.y1 - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        cv::rectangle(image,
                      cv::Rect(cv::Point(x, y),
                               cv::Size(label_size.width, label_size.height + baseLine)),
                      color,
                      -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), thickness);
    }
    cv::imwrite(save_path, image);
}

int image_demo(PicoDet &detector, const char *imagepath,
               int has_postprocess = 0) {
    std::vector<cv::String> filenames;
    cv::glob(imagepath, filenames, false);
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
    const char *param_model_path = "../models/picodet.param";
    const char *bin_model_path = "../models/picodet.bin";
    PicoDet detector =
            PicoDet(param_model_path, bin_model_path, 320, 320, 0.45, 0.3);
    const char *images = "../R.jpg";
    image_demo(detector, images, false);
}