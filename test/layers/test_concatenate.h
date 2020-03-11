/**
 * @brief contains all activation functions
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 * @author Nils Weinhardt (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef TEST_CONCATENATE_H
#define TEST_CONCATENATE_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/concatenate.h"
#include "../utility_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int32_t test_concatenate_2D_1();
int32_t test_concatenate_2D_2();
int32_t test_concatenate_naive_dim_0();
int32_t test_concatenate_naive_dim_1();
int32_t test_concatenate_naive_dim_2();

#endif //TEST_CONCATENATE_H
