//
// Created by junga on 10.01.20.
//

#ifndef PICO_CNN_BATCH_NORMALIZATION_H
#define PICO_CNN_BATCH_NORMALIZATION_H

#include "../parameters.h"
#include <stdint.h>
#include <math.h>

void batch_normalization_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                               fp_t* output_channel, const fp_t gamma, const fp_t beta, const fp_t mean,
                               const fp_t variance, const fp_t epsilon);

#endif //PICO_CNN_BATCH_NORMALIZATION_H
