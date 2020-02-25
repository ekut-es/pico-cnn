/**
 * @brief contains all poolings
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef POOLING_H
#define POOLING_H

#include "../parameters.h"
#include "../utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

/**
 * @brief applies max pooling of kernel_size to input_channel
 *
 * @param input_channel
 * @param output_channel
 * @param kernel_size
 * @param stride
 */
void max_pooling1d_naive(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel,
                         const uint16_t kernel_size, const uint16_t stride);

/**
 * @brief Extends the input channel with the given padding and passes the extended input channel to max_pooling1d_naive
 *
 * @param input_channel
 * @param input_width
 * @param output_channel
 * @param kernel_size
 * @param stride
 * @param padding
 */
void max_pooling1d_naive_padded(const fp_t* input_channel, const uint16_t input_width,
                                fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                                const int* padding);

/**
 * @brief applies max pooling of kernel_size x kernel_size to input_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (((height-kernel_size)/stride+1) x ((width-kernel_size)/stride+1))
 * @param kernel_size
 * @param stride
 */
void max_pooling2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                         fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride);

/**
 * @brief Extends the input channel with the given padding and passes the extended input channel to max_pooling2d_naive
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (((height+padding[0]+padding[2]-kernel_size)/stride+1) x ((width+padding[1]+padding[3]-kernel_size)/stride+1))
 * @param kernel_size
 * @param stride
 * @param padding integer array of length 4 containing the number of padding pixels for each dimension (x1_begin, x2_begin, x1_end, x2_end)
 */
void max_pooling2d_naive_padded(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                                fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                                const int* padding);

/**
 * @brief applies average pooling of kernel_size x kernel_size to input_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void average_pooling2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                             fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                             fp_t bias, const uint16_t count_include_pad);

/**
 * @brief Extends the input channel with the given padding and passes the extended input channel to average_pooling2d_naive
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel
 * @param kernel_size
 * @param bias
 */
void average_pooling2d_naive_padded(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                                    fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                                    fp_t bias, const int* padding, const uint16_t count_include_pad);

/**
 *
 * @param input_channel
 * @param input_width
 * @param output_channel
 * @param kernel_size
 * @param stride
 * @param bias
 */
void average_pooling1d_naive(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel,
                             const uint16_t kernel_size, const uint16_t stride, fp_t bias,
                             const uint16_t count_include_pad);

/**
 *
 * @param input_channel
 * @param input_width
 * @param output_channel
 * @param kernel_size
 * @param stride
 * @param bias
 * @param padding
 */
void average_pooling1d_naive_padded(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel,
                                    const uint16_t kernel_size, const uint16_t stride, fp_t bias, const int* padding,
                                    const uint16_t count_include_pad);



void global_average_pooling2d_naive(const fp_t* input_channel, const uint16_t input_height,
                                    const uint16_t input_width, fp_t* output_channel);

void global_max_pooling2d_naive(const fp_t* input_channel, const uint16_t input_height,
                                const uint16_t input_width, fp_t* output_channel);

void pad_2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                  fp_t* output_channel, const int* padding, fp_t initializer);

void pad_1d_naive(const fp_t* input_channel, const uint16_t width,
                  fp_t* output_channel, const int* padding, fp_t initializer);

#endif // POOLING_H
