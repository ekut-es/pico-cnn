#ifndef TEST_CONVOLUTION_H
#define TEST_CONVOLUTION_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/convolution.h"
#include "../utility_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int test_convolution1d_naive();
int test_convolution2d_naive();
int test_convolution2d_naive_1();
int test_convolution2d_naive_2(); 
int test_add_channel2d_naive();
#endif // TEST_CONVOLUTION_H
