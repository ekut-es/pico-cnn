/**
 * @brief contains all activation functions
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 * @author Nils Weinhardt (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef TEST_ACTIVATION_FUNCTION_H
#define TEST_ACTIVATION_FUNCTION_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/activation_function.h"
#include "../utility_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int32_t test_relu_naive();

int32_t test_leaky_relu_naive();

int32_t test_parametrized_relu_naive();

int32_t test_sigmoid_naive();

int32_t test_softmax_naive();

int32_t test_local_response_normalization_naive();

#endif //TEST_ACTIVATION_FUNCTION_H
