#ifndef CNN_IP_H
#define CNN_IP_H

#include "fixpoint.h"

#define USE_FLOAT

#ifdef USE_FIXED_POINT
typedef cnn::Fixed16 data_t;
typedef cnn::Fixed16 weight_t;
typedef cnn::Fixed16 acc_t;
#else
typedef float data_t;
typedef float weight_t;
typedef float acc_t;
#endif

void cnn_top(
    const data_t input_img[24][24][3],
    data_t output_logits[10]
);

#endif 