//
// Created by AC on 2023/8/30.
//

#include "picodet.h"


#ifndef CAPI
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define API_EXPORT __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
#   define API_EXPORT __attribute__ ((visibility ("default")))
# endif
#endif


#ifdef __cplusplus
extern "C" {
#endif
int init_model(void **model, int input_width, int input_height, float score_threshold, float nms_threshold);

int image_predict_file(void *model, const char *image_pat);

int image_predict_buffer(void *model, void *buffer, void *out_buffer, int w, int h);


#ifdef __cplusplus
}
#endif