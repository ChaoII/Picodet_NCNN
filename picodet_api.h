//
// Created by AC on 2023/8/30.
//

#include "picodet.h"
#include "utils.h"

#if defined(_WIN32)
#ifndef CAPI
#define CAPI
#endif
#ifdef CAPI
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif  // CAPI
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#define model_handle_t void *

#ifdef __cplusplus
extern "C" {
#endif

///
/// 初始化模型句柄
/// \param model_handle 模型对象句柄
/// \param input_width 模型输入的宽度和高度，固定宽高320
/// \param input_height 模型输入图像高度，固定高度320
/// \param score_threshold score阈值，根据得分标注数据，如果画面中框多，框错，请酌情调高阈值
/// \param nms_threshold 非最大抑制阈值，当有框重叠时酌情调整，当阈值大于该值时框会被抑制
API_EXPORT void init_model(model_handle_t*model_handle, int input_width, int input_height,
                           float score_threshold, float nms_threshold = 0.45f);
/// 通过磁盘中的文件预测图片中人的数量
/// \param model_handle 模型对象句柄
/// \param image_path 图像路径
/// \param out_buffer 输出buffer，模型绘制框后绘制
/// \param save_file_name 保存图像文件路径，如果传nullptr则不保存图像
/// \return 图片中行人的数量
API_EXPORT int image_predict_file(model_handle_t model_handle, const char *image_path,
                                  void *out_buffer, const char *save_file_name = nullptr);

/// 通过内存中的图像字节预测图片中人的数量
/// \param model_handle 通过磁盘中的文件预测图片中人的数量
/// \param in_buffer 输入图像buffer
/// \param out_buffer 输出图像buffer
/// \param w 图像宽
/// \param h 图像高
/// \param save_file_name 保存图像文件路径，如果传nullptr则不保存图像
/// \return 图片中行人的数量
API_EXPORT int image_predict_buffer(model_handle_t model_handle, void *in_buffer, void *out_buffer,
                                    int w, int h, const char *save_file_name = nullptr);

/// 释放模型句柄，释放内存
/// \param model_handle
API_EXPORT void free_model(model_handle_t model_handle);


#ifdef __cplusplus
}
#endif