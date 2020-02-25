/**
 * @brief provides utility functions used in the CNN
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef UTILS_H
#define UTILS_H

#include "parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

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


#endif // UTILS_H
