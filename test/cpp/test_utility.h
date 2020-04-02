#ifndef PICO_CNN_TEST_UTILITY_H
#define PICO_CNN_TEST_UTILITY_H

#include <cstdint>
#include <cstring>
#include <cmath>

#include "../../pico-cnn/parameters.h"

int32_t floats_almost_equal(fp_t f1, fp_t f2, fp_t err);

static inline fp_t urand(fp_t min, fp_t max);

#endif //PICO_CNN_TEST_UTILITY_H
