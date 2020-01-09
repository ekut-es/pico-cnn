#ifndef TEST_ACTIVATION_FUNCTION_H
#define TEST_ACTIVATION_FUNCTION_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/activation_function.h"
#include "../utility_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int test_relu_naive();

int test_leaky_relu_naive();

int test_parametrized_relu_naive();

int test_sigmoid_naive();

int test_softmax_naive();

int test_local_response_normalization_naive();

#endif //TEST_ACTIVATION_FUNCTION_H
