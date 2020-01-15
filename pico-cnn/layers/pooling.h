/**
 * @brief contains all poolings
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef POOLING_H
#define POOLING_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif

#ifdef ARM_NEON
#include "arm_neon.h"
#include <float.h>
#endif

/**
 * @brief Extends input_channel with padding by copying the data into a bigger array.
 *
 * @param input_channel
 * @param height
 * @param width
 * @param extended_input
 * @param padding
 */
void extend_2d_input_with_padding(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                                  fp_t** extended_input, const int* padding, fp_t initializer);

/**
 * @brief Extends input_channel with padding by copying the data into a bigger array.
 *
 * @param input_channel
 * @param width
 * @param extended_input
 * @param padding
 */
void extend_1d_input_with_padding(const fp_t* input_channel, const uint16_t width,
                                  fp_t** extended_input, const int* padding, fp_t initializer);

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



void global_average_pooling2d_naive(const fp_t* input_channel, const uint16_t input_width,
                                    const uint16_t input_height, fp_t* output_channel);

#ifdef FIXED16
/**
 * @brief applies max pooling of kernel_size x kernel_size to input_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height/kernel_size x width/kernel_size)
 * @param kernel_size
 * @param stride
*/
void max_pooling2d_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width,
                                 fixed16_t* output_channel, const uint16_t kernel_size, const uint16_t stride);

/**
 * @brief TODO
 */
void max_pooling2d_cpu_2x2_s2_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width,
                                      fixed16_t* output_channel);

#endif // FIXED16

#ifdef ARM_NEON
/**
 * @brief applies max pooling of kernel_size x kernel_size to input_channel
 *
 * kernel_size = 2
 * stride = 2
 *
 * @param input_channel (height x width)
 * @param output_channel (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void max_pooling2d_cpu_2x2_s2(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                              fp_t* output_channel);


/**
 * @brief applies max pooling of kernel_size x kernel_size to input_channel
 *
 * kernel_size = 3
 * stride = 2
 *
 * @param input_channel (height x width)
 * @param output_channel (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void max_pooling2d_cpu_3x3_s2(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                              fp_t* output_channel);

#endif // ARM_NEON

#endif // POOLING_H
