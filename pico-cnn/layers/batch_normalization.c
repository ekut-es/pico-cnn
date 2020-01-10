//
// Created by junga on 10.01.20.
//

#include "batch_normalization.h"

void batch_normalization_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                               fp_t* output_channel, const fp_t gamma, const fp_t beta, const fp_t mean,
                               const fp_t variance, const fp_t epsilon) {

    uint16_t channel_idx;

    for(channel_idx = 0; channel_idx < height*width; channel_idx++){
        output_channel[channel_idx] = gamma * (input_channel[channel_idx] - mean) / sqrtf(variance + epsilon) + beta;
    }

}