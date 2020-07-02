/**
 * @brief contains all activation functions
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 * @author Nils Weinhardt (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef TEST_CONVOLUTION_H
#define TEST_CONVOLUTION_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/convolution.h"
#include "../utility_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int32_t test_convolution1d_naive();
int32_t test_convolution2d_naive();
int32_t test_convolution2d_naive_1();
int32_t test_convolution2d_naive_2();
int32_t test_convolution2d_naive_3(); // with non-square kernel
int32_t test_convolution2d_naive_4(); // with non-square kernel
int32_t test_convolution2d_naive_5(); // with non-square kernel
int32_t test_add_channel2d_naive();
int32_t test_convolution2d_naive_6();
int32_t test_convolution2d_naive_7();
int32_t test_convolution2d_naive_8();

#endif // TEST_CONVOLUTION_H
