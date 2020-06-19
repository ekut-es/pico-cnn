//
// Created by junga on 15.05.20.
//

#include "batch_normalization.h"

pico_cnn::naive::BatchNormalization::BatchNormalization(std::string name, uint32_t id, pico_cnn::op_type op,
                                                        Tensor *gammas, Tensor *betas, Tensor *means, Tensor *variances,
                                                        fp_t epsilon) : Layer(name, id, op) {
    gammas_ = gammas;
    betas_ = betas;
    means_ = means;
    variances_ = variances;
    epsilon_ = epsilon;
}

void pico_cnn::naive::BatchNormalization::run(pico_cnn::naive::Tensor *input, pico_cnn::naive::Tensor *output) {
    if (input->num_dimensions() != 4) {
        PRINT_ERROR_AND_DIE("Not implemented for Tensor with num_dims: " << input->num_dimensions());
    }

    uint32_t num_batches = input->num_batches();
    if (num_batches != 1)
        PRINT_ERROR_AND_DIE("Number of batches != 1. BatchNormalization not implemented for this.")

    uint32_t num_input_channels = input->num_channels();

    for (uint32_t channel = 0; channel < num_input_channels; channel++) {
        this->normalize(input, output, channel);
    }

}

void pico_cnn::naive::BatchNormalization::normalize(pico_cnn::naive::Tensor *input, pico_cnn::naive::Tensor *output, uint32_t channel) {

    uint32_t num_channels = input->num_channels();
    uint32_t input_height = input->height();
    uint32_t input_width = input->width();

    fp_t gamma = gammas_->access(channel);
    fp_t beta = betas_->access(channel);
    fp_t mean = means_->access(channel);
    fp_t variance = variances_->access(channel);

    for (uint32_t row = 0; row < input_height; row++) {
        for (uint32_t col = 0; col < input_width; col++) {
            output->access(0, channel, row, col, num_channels, input_height, input_width) =
                    gamma * (input->access(0, channel, row, col, num_channels, input_height, input_width) - mean) /
                    sqrtf(variance + epsilon_) + beta;
        }
    }
}
