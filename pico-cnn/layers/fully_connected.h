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

#endif // FULLY_CONNECTED_H
