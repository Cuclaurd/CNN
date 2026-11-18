#ifndef CNN_IP_H
#define CNN_IP_H


// #define USE_FIXED_POINT

#ifdef USE_FIXED_POINT
///.. à faire apres ahah

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