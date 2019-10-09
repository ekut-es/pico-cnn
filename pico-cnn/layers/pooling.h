/**
 * @brief contains all poolings
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef POOLING_H
#define POOLING_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif

#ifdef ARM_NEON
#include "arm_neon.h"
#include <float.h>
#endif

/**
 * @brief applies max pooling of kernel_size to input_channel
 *
 * @param input_channel
 * @param output_channel
 * @param kernel_size
 * @param stride
 */
void max_pooling1d_naive(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride);


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
void max_pooling2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride);

/**
 * @brief applies average pooling of kernel_size x kernel_size to input_channel
 *
 * @param input_channel (height x width)
 * @param height
 * @param width
 * @param output_channel (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void average_pooling2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel, const uint16_t kernel_size, fp_t bias);


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
void max_pooling2d_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel, const uint16_t kernel_size, const uint16_t stride);

/**
 * @brief TODO
 */
void max_pooling2d_cpu_2x2_s2_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width, fixed16_t* output_channel);

#endif

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
void max_pooling2d_cpu_2x2_s2(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel);


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
void max_pooling2d_cpu_3x3_s2(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel);
#endif

#endif // POOLING_H
