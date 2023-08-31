//
// Created by AC on 2023/8/31.
//

#include "utils.h"
#include "filesystem"

namespace fs = std::filesystem;

bool Utils::has_image_extension(const std::string &filename) {
    fs::path file_path(filename);
    if (!file_path.has_extension()) {
        std::cerr << "is not a file, file extension must in ['.jpg','.bmp','png'] or other supported image format";
        return false;
    }
    std::string extension_string = file_path.extension().string();
    tolower_string(extension_string);
    if (allowed_extensions.count(extension_string)) {
        return true;
    } else {
        std::cerr << "The file does not have an allowed extension." << std::endl;
        return false;
    }
}


void Utils::tolower_string(std::string &str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
}

