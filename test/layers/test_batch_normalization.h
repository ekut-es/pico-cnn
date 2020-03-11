/**
 * @brief contains all activation functions
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 * @author Nils Weinhardt (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_TEST_BATCH_NORMALIZATION_H
#define PICO_CNN_TEST_BATCH_NORMALIZATION_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/batch_normalization.h"
#include "../utility_functions.h"

#include <stdlib.h>

int32_t test_batch_normalization_naive_1();
int32_t test_batch_normalization_naive_2();

#endif //PICO_CNN_TEST_BATCH_NORMALIZATION_H
