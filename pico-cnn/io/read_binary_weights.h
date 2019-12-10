//
// Created by junga on 04.07.19.
//

#ifndef PICO_CNN_READ_BINARY_WEIGHTS_H
#define PICO_CNN_READ_BINARY_WEIGHTS_H

#include "../parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int read_binary_weights(const char* path_to_weights_file, fp_t**** kernels, fp_t*** biases);

#endif //PICO_CNN_READ_BINARY_WEIGHTS_H
