/**
 * @brief provides functions for reading reference input and ouput data
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_READ_BINARY_REFERENCE_DATA_H
#define PICO_CNN_READ_BINARY_REFERENCE_DATA_H

#include "../parameters.h"
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include "../tensor.h"

int32_t read_binary_reference_input_data(const char* path_to_sample_data, pico_cnn::naive::Tensor **input_tensor);

int32_t read_binary_reference_output_data(const char* path_to_sample_data, pico_cnn::naive::Tensor **output_tensor);

#endif //PICO_CNN_READ_BINARY_REFERENCE_DATA_H
