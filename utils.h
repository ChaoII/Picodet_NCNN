//
// Created by AC on 2023/8/31.
//

#pragma once

#include "iostream"
#include "vector"
#include "set"


class Utils {

public:
    static bool has_image_extension(const std::string &filename);

    static void tolower_string(std::string &str);

private:
    inline static std::set<std::string> allowed_extensions = {"jpg", "bmp", "png"};

};


