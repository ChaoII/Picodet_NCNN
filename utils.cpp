//
// Created by AC on 2023/8/31.
//

#include <algorithm>
#include "utils.h"


bool Utils::has_image_extension(const std::string &filename) {
    size_t pos = filename.rfind('.');
    if (pos != std::string::npos) {
        std::string fileExt = filename.substr(pos + 1);
        tolower_string(fileExt);
        if (allowed_extensions.count(fileExt)) {
            return true;
        } else {
            std::cerr << "The file does not have an allowed extension." << std::endl;
            return false;
        }
    }
    return false;
}


void Utils::tolower_string(std::string &str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
}

