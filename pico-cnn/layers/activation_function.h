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

#endif // ACTIVATION_FUNCTION_H
