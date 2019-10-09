/**
 * @brief contains the fully connected layer
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "../parameters.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif

#ifdef ARM_NEON
#include "arm_neon.h"
#endif


/**
 * @brief implementation of fully connected layer
 *
 * @param input_channel (1 x input_width)
 * @param input_width
 * @param output_channel (1 x output_width)
 * @param output_width
 * @param kernel
 * @param bias
 */
void fully_connected_naive(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel, const uint16_t output_width, const fp_t* kernel, const fp_t* bias) {

    int i, j;
    for(i = 0; i < output_width; i++) {

        fp_t pixel = 0.0;

        for(j = 0; j < input_width; j++) {
            // takes each output_width'nd element
            pixel += input_channel[j] * kernel[j*output_width+i];
        }

        pixel += bias[i];
        output_channel[i] = pixel;
    }
}

#ifdef FIXED16
/**
 * @brief implementation of fully connected layer
 *
 * @param input_channel (1 x input_width)
 * @param input_width
 * @param output_channel (1 x output_width)
 * @param output_width
 * @param kernel
 * @param bias
 */
void fully_connected_naive_fixed16(const fixed16_t* input_channel, const uint16_t input_width, fixed16_t* output_channel, const uint16_t output_width, const fixed16_t* kernel, const fixed16_t* bias) {

    int i, j;
    for(i = 0; i < output_width; i++) {

        fp_t pixel = 0.0;

        for(j = 0; j < input_width; j++) {
            // takes each output_width'nd element
            pixel = add_fixed16(pixel, mul_fixed16(input_channel[j], kernel[j*output_width+i]));
        }

        pixel = add_fixed16(pixel, bias[i]);
        output_channel[i] = pixel;
    }
}

#endif

#ifdef ARM_NEON
/**
 * @brief resturctes kernel for fully connected layer such that a vectorized
 * access is possible
 *
 * @param original_kernel
 * @param input_width of fully connected layer
 * @param output_width of fully connected layer
 */
void restructure_fully_connected_kernel(fp_t** kernel, const uint16_t input_width, const uint16_t output_width) {
    fp_t* new_kernel = (fp_t*) malloc(input_width*output_width*sizeof(fp_t));

    uint32_t i,j;

    for(i = 0 ; i < output_width; i++) {
        for(j = 0; j < input_width; j++) {
            new_kernel[i*input_width+j] = (*kernel)[j*output_width+i];
        }
    }

    free(*kernel);
    *kernel = new_kernel;
}

/**
 * @brief implementation of fully connected layer optimzed for CPU
 *
 * @param input_channel (1 x input_width)
 * @param input_width
 * @param output_channel (1 x output_width)
 * @param output_width
 * @param kernel
 * @param bias
 */
void fully_connected_cpu(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel, const uint16_t output_width, const fp_t* kernel, const fp_t* bias, const uint32_t from, const uint32_t to) {

    float32x4_t kernel_0 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t kernel_1 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t kernel_2 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t kernel_3 = {0.0, 0.0, 0.0, 0.0};

    float32x4_t input_channel_0;
    float32x4_t input_channel_1;
    float32x4_t input_channel_2;
    float32x4_t input_channel_3;


    uint32_t i, j;
    for(i = from; i < to; i++) {

        fp_t pixel = 0.0;

        for(j = 0; j < input_width-BLOCK_SIZE; j+=BLOCK_SIZE) {
            // load kernel into vectors
            kernel_0 = vld1q_f32(kernel+i*input_width+j);
            kernel_1 = vld1q_f32(kernel+i*input_width+j+4);
            kernel_2 = vld1q_f32(kernel+i*input_width+j+8);
            kernel_3 = vld1q_f32(kernel+i*input_width+j+12);

            // load input channel into vectors
            input_channel_0 = vld1q_f32(input_channel+j);
            input_channel_1 = vld1q_f32(input_channel+j+4);
            input_channel_2 = vld1q_f32(input_channel+j+8);
            input_channel_3 = vld1q_f32(input_channel+j+12);

            // apply kernel
            input_channel_0 = vmulq_f32(input_channel_0, kernel_0);
            input_channel_1 = vmulq_f32(input_channel_1, kernel_1);
            input_channel_2 = vmulq_f32(input_channel_2, kernel_2);
            input_channel_3 = vmulq_f32(input_channel_3, kernel_3);

            // sum up
            input_channel_0 = vaddq_f32(input_channel_0, input_channel_1);
            input_channel_0 = vaddq_f32(input_channel_0, input_channel_2);
            input_channel_0 = vaddq_f32(input_channel_0, input_channel_3);

            // store in pixel
            pixel += vgetq_lane_f32(input_channel_0, 0) + vgetq_lane_f32(input_channel_0, 1) + vgetq_lane_f32(input_channel_0, 2) + vgetq_lane_f32(input_channel_0, 3);
        }

        // residual pixels
        for(j = j; j < input_width; j++) {
            pixel += input_channel[j] * kernel[i*input_width+j];
        }

        pixel += bias[i];
        output_channel[i] = pixel;
    }
}
#endif


#endif // FULLY_CONNECTED_H
