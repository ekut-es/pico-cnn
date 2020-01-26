#ifndef CONCATENATE_H
#define CONCATENATE_H

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

/*  @brief concatenates 1D channels into a single (2D) channel
 *         output channel needs to be of size width  * num_inputs * sizeof(fp_t)
 */
 void concatenate_1D(const fp_t** input_channels, const uint16_t width, const uint16_t num_inputs, fp_t* output_channel);

/* @brief concatenates 2D channels into a single (2D) channel
 * the output channel needs to be of size width * height * num_inputs * sizeof(fp_t)
 * @param dimension the dimension on which to concatenate, 0: height dimension, 1: width dimension
 */
 void concatenate_2D(const fp_t** input_channels, const uint16_t width, const uint16_t height,
                     const uint16_t dimension, const uint16_t num_inputs, fp_t* output_channel);

/* @brief concatenates multiple 3D inputs (lists of channels) into a single (3D) output (list of 2D channels).
 * the output channel need to be of size width * height * num_input_channels * num_inputs * sizeof(fp_t)
 * the size of the output channels varies with the dimension
 * @param channel_width
*/
void concatenate_3D(const fp_t*** inputs, const uint16_t channel_width, const uint16_t channel_height,
                    const uint16_t dimension, const uint16_t num_inputs, const uint16_t num_input_channels,
                    fp_t** output_channels);


#endif //CONCATENATE_H
