//
// Created by junga on 27.08.19.
//

#ifndef PICO_CNN_READ_BINARY_SAMPLE_DATA_H
#define PICO_CNN_READ_BINARY_SAMPLE_DATA_H

#include "../parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int read_binary_sample_input_data(const char* path_to_sample_data, fp_t*** input);

int read_binary_sample_output_data(const char* path_to_sample_data, fp_t** output);

#endif //PICO_CNN_READ_BINARY_SAMPLE_DATA_H
