/**
 * @brief contains all convolutions
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../parameters.h"
#include "../utils.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif

#ifdef ARM_NEON
#include "arm_neon.h"
#endif

/**
 * @brief performs a 1D convolution on input_channel with kernel and stores the
 * result to output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel
 * @param kernel
 * @param kernel_size
 * @param stride
 * @param padding (0 means valid, > 0 zeros will be added to the edge). Padding for both sides
 * @param dilation
 * @param bias
 */
void convolution1d_naive(const fp_t* input_channel, const int input_size, fp_t* output_channel, const fp_t* kernel,
                         const int kernel_size, const int stride, const int padding, const fp_t bias);


/**
 * @brief performs a 2D convolution on input_channel with kernel and stores the
 * result to output_channel. Legacy version, assumes square, uneven kernel and same stride 
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_size x kernel_size)
 * @param kernel_size
 * @param stride
 * @param padding (0 means valid, > 0 zeros will be added to the edge)
 * @param bias
 */
void convolution2d_naive_legacy(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                                fp_t* output_channel, const fp_t* kernel, const uint16_t kernel_size,
                                const uint16_t stride, const uint16_t padding, const fp_t bias);


/**
 * @brief performs a 2D convolution on input_channel with kernel and stores the
 * result to output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_height * kernel_width)
 * @param kernel_height needs to be uneven
 * @param kernel_width needs to be even
 * @param stride_height stride in height direction
 * @param stride_width stride in width direction
 * @param bias
 */
void convolution2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel,
                         const fp_t* kernel, const uint16_t kernel_height, const uint16_t kernel_width,
                         const uint16_t stride_height, const uint16_t stride_width, const fp_t bias);


/**
 * @brief performs a 2D convolution on input_channel with kernel and stores the
 * result to output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_height * kernel_width)
 * @param kernel_height needs to be uneven
 * @param kernel_width needs to be even
 * @param stride_height stride in height direction
 * @param stride_width stride in width direction
 * @param padding integer array with length four, specifying padding
 *        padding[0], padding[2]: Padding at start, end of height dimension
 *        padding[1], padding[3]: Padding at start, end of width dimension
 * @param bias
 */
void convolution2d_padding_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel,
                                 const fp_t* kernel, const uint16_t kernel_height, const uint16_t kernel_width,
                                 const uint16_t stride_height, const uint16_t stride_width,
                                 const int* padding, const fp_t bias);

/**
 * @brief adds channel_a and channel_b pixel by pixel and stores result in channel_a
 *
 * @param channel_a (height x width)
 * @param channel_b (height x width)
 * @param height
 * @param width
 */
void add_channel2d_naive(fp_t* channel_a, const fp_t* channel_b, const uint16_t height, const uint16_t width);

#ifdef ARM_NEON
/**
 * @brief performs an CPU optimized 2D convolution on input_channel with a
 * kernel 3x3 and stores the result to output_channel
 *
 * stride = 1
 * padding = valid => channel shrinks by 1 pixel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height-1 x width-1)
 * @param kernel (3x3)
 * @param bias
 */
void convolution2d_cpu_3x3_s1_valid(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias);

/**
 * @brief performs an CPU optimized 2D convolution on input_channel with a
 * kernel 3x3 and stores the result to output_channel
 *
 * stride = 1
 * padding = same
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 * @param kernel (3x3)
 * @param bias
 */
void convolution2d_cpu_3x3_s1_same(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias);

/**
 * @brief performs an CPU optimized 2D convolution on input_channel with a
 * kernel 5x5 and stores the result to output_channel
 *
 * stride = 1
 * padding = valid => channel shrinks by 2 pixels
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height-2 x width-2)
 * @param kernel (5x5)
 * @param bias
 */
void convolution2d_cpu_5x5_s1_valid(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias);

/**
 * @brief performs an CPU optimized 2D convolution on input_channel with a
 * kernel 5x5 and stores the result to output_channel
 *
 * stride = 1
 * padding = same
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height x width)
 * @param kernel (5x5)
 * @param bias
 */
void convolution2d_cpu_5x5_s1_same(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias);

/**
 * @brief performs an CPU optimized 2D convolution on input_channel with a
 * kernel 11x11 and stores the result to output_channel
 *
 * stride = 4
 * padding = valid
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel
 * @param kernel (11x11)
 * @param bias
 */
void convolution2d_cpu_11x11_s4_valid(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const fp_t* kernel, const fp_t bias);

/**
 * @brief adds channel_a and channel_b pixel by pixel and stores result in channel_a optimized for CPU
 *
 * @param channel_a (height x width)
 * @param channel_b (height x width)
 * @param height
 * @param width
 */
void add_channel2d_cpu(fp_t* channel_a, const fp_t* channel_b, const uint16_t height, const uint16_t width);
#endif //ARM_NEON


#ifdef FIXED16
/**
 * @brief performs a 2D convolution on input_channel with kernel and stores the
 * result to output_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_size x kernel_size)
 * @param kernel_size
 * @param stride
 * @param padding (0 means valid, > 0 zeros will be added to the edge)
 * @param bias
 */
void convolution2d_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel, const fixed16_t* kernel, const uint16_t kernel_size, const uint16_t stride, const uint16_t padding, const fixed16_t bias);

/**
 * @brief TODO
 */
void convolution2d_cpu_5x5_s1_valid_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel, const fixed16_t* kernel, const fixed16_t bias);

/**
 * @brief adds channel_a and channel_b pixel by pixel and stores result in channel_a
 *
 * @param channel_a (height x width)
 * @param channel_b (height x width)
 * @param height
 * @param width
 */
void add_channel2d_naive_fixed16(fixed16_t* channel_a, const fixed16_t* channel_b, const uint16_t height, const uint16_t width);

/**
 * @brief TODO
 */
void add_channel2d_cpu_fixed16(fixed16_t* channel_a, const fixed16_t* channel_b, const uint16_t height, const uint16_t width);
#endif // FIXED16

#endif // CONVOLUTION_H
