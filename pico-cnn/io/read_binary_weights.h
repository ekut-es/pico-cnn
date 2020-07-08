/**
 * @brief provides the function for reading the binary weights file
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_READ_BINARY_WEIGHTS_H
#define PICO_CNN_READ_BINARY_WEIGHTS_H

#include "../parameters.h"
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include "../tensor.h"

int32_t read_binary_weights(const char* path_to_weights_file, pico_cnn::naive::Tensor ***kernels, pico_cnn::naive::Tensor ***biases);

#endif //PICO_CNN_READ_BINARY_WEIGHTS_H
