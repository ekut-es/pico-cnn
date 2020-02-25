//
// Created by junga on 27.08.19.
//

#ifndef PICO_CNN_READ_BINARY_REFERENCE_DATA_H
#define PICO_CNN_READ_BINARY_REFERENCE_DATA_H

#include "../parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

//#define DEBUG

int read_binary_reference_input_data(const char* path_to_sample_data, fp_t*** input);

int read_binary_reference_output_data(const char* path_to_sample_data, fp_t** output);

#endif //PICO_CNN_READ_BINARY_REFERENCE_DATA_H
