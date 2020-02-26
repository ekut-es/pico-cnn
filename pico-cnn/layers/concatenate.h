/**
 * @brief contains concatenate operation
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 * @author Nils Weinhardt (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef CONCATENATE_H
#define CONCATENATE_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

/**
 * @brief Concatenates 1D channels into a single (2D) channel.
 *        output_channel needs to be of size width * num_inputs
 * @param input_channels
 * @param width
 * @param num_inputs
 * @param output_channel
 */
void concatenate_1D(fp_t** input_channels, const uint16_t width, const uint16_t num_inputs, fp_t* output_channel);

/**
 * @brief Concatenates 2D channels into a single (2D) channel
 *        output_channel needs to be of size width * height * num_inputs
 *
 * @param input_channels
 * @param width
 * @param height
 * @param dimension 0: height dimension, 1: width dimension
 * @param num_inputs
 * @param output_channel
 */
void concatenate_2D(fp_t** input_channels, const uint16_t width, const uint16_t height,
                     const uint16_t dimension, const uint16_t num_inputs, fp_t* output_channel);

/**
 * @brief Concatenates multiple inputs consisting of multiple channels of 2D data
 *        into a single output containing all data. The dimension on which to concatenate
 *        is specified with the dimension argument.
 *
 *        E.g. inputs with the following shapes can be concatenated:
 *
 *        Along dimension = 0
 *        input_shape = {
 *                          {32, 35, 35},
 *                          {96, 35, 35},
 *                          {64, 35, 35},
 *                          {64, 35, 35}
 *                      }
 *
 *        Along dimension = 1
 *        input_shape = {
 *                          {32, 10, 35},
 *                          {32, 15, 35},
 *                          {32,  5, 35},
 *                          {32, 35, 35}
 *                      }
 *
 *        Along dimension = 2
 *        input_shape = {
 *                          {32, 35, 10},
 *                          {32, 35, 15},
 *                          {32, 35,  5},
 *                          {32, 35, 35}
 *                      }
 *
 * @param inputs
 * @param input_shape
 * @param dimension
 * @param num_inputs
 * @param output_channels
 */
void concatenate_naive(fp_t*** inputs, const uint16_t** input_shape, const uint16_t dimension,
                       const uint16_t num_inputs, fp_t** output_channels);


#endif //CONCATENATE_H
