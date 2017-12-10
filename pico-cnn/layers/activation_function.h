/** 
 * @brief contains all activation functions 
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "../parameters.h"
#include <stdint.h>
#include <math.h>


/**
 * @brief applies tanh(x) to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void tanh_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    for(i = 0; i < height*width; i++) {
        new_image[i] = tanhf(original_image[i]);
    }
}

/**
 * @brief applies relu(x) to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void relu_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    for(i = 0; i < height*width; i++) {
        new_image[i] = (original_image[i] < 0.0) ? 0.0 : original_image[i];
    }
}

/**
 * @brief applies softmax to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void softmax_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    fp_t denominator = 0.0;

    for(i = 0; i < height*width; i++) {
        denominator += expf(original_image[i]);
    }

    for(i = 0; i < height*width; i++) {
        new_image[i] = expf(original_image[i]) / denominator;
    }
}

/**
 * @brief performs a local response normalization (across channels) on original 
 * image and stores the result in new_image
 *
 * Formula (Paper):
 * https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn/252343#252343
 * Formula (Implemented):
 * http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param depth
 * @param new_image
 * @param alpha
 * @param beta
 * @param n
 */
void local_response_normalization_naive(fp_t** original_image, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** new_image, const fp_t alpha, const fp_t beta, const uint16_t n) {
    
    int32_t channel, row, column, i;
    int32_t from;
    int32_t to;

    fp_t sum;

    for(channel = 0; channel < depth; channel++) {
        from = MAX(0,channel-(n/2));
        to = MIN(depth-1,channel+(n/2));

        for(row = 0; row < height; row++) {
            for(column = 0; column < width; column++) {

                sum = 0.0;

                for(i = from; i <= to; i++) {
                    sum += powf(original_image[i][row*width+column], 2);
                }

                new_image[channel][row*width+column] = original_image[channel][row*width+column] / powf((1+(alpha/n)*sum),beta);
            }
        }
    }
}

#ifdef __aarch64__ 
/**
 * @brief applies relu(x) to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void relu_cpu(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    for(i = 0; i < height*width; i++) {
        new_image[i] = (original_image[i] < 0.0) ? 0.0 : original_image[i];
    }
}
#endif

#endif // ACTIVATION_FUNCTION_H
