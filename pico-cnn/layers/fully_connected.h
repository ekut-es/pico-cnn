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
void fully_connected_naive(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel, const uint16_t output_width, const fp_t* kernel, const fp_t* bias);

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
void fully_connected_naive_fixed16(const fixed16_t* input_channel, const uint16_t input_width, fixed16_t* output_channel, const uint16_t output_width, const fixed16_t* kernel, const fixed16_t* bias);

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
void restructure_fully_connected_kernel(fp_t** kernel, const uint16_t input_width, const uint16_t output_width);

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
void fully_connected_cpu(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel, const uint16_t output_width, const fp_t* kernel, const fp_t* bias, const uint32_t from, const uint32_t to);
#endif


#endif // FULLY_CONNECTED_H
