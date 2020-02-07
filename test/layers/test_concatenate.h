#ifndef TEST_CONCATENATE_H
#define TEST_CONCATENATE_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/concatenate.h"
#include "../utility_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int test_concatenate_2D_1();
int test_concatenate_2D_2();
int test_concatenate_3D_1();
int test_concatenate_3D_2();
int test_concatenate_3D_3();
int test_concatenate_naive_dim_0();
int test_concatenate_naive_dim_1();
int test_concatenate_naive_dim_2();

#endif //TEST_CONCATENATE_H