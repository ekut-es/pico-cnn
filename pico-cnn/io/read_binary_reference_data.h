/**
 * @brief provides functions for reading reference input and ouput data
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_READ_BINARY_REFERENCE_DATA_H
#define PICO_CNN_READ_BINARY_REFERENCE_DATA_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define DEBUG

int read_binary_reference_input_data(const char* path_to_sample_data, fp_t*** input);

int read_binary_reference_output_data(const char* path_to_sample_data, fp_t** output);

#endif //PICO_CNN_READ_BINARY_REFERENCE_DATA_H
