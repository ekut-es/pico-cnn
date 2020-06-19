//
// Created by junga on 03.04.20.
//

#include "lrn.h"
namespace pico_cnn {
    namespace naive {

        LRN::LRN(std::string name, uint32_t id, op_type op, const fp_t alpha, const fp_t beta, const uint16_t n) :
                ActivationFunction(name, id, op), alpha_(alpha), beta_(beta), n_(n) {

        }

        LRN::~LRN() {

        }

        void LRN::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void LRN::activate(Tensor *input, Tensor *output) {
            if (input->num_dimensions() == 4) {
                int32_t i, num_batches, num_channels, height, width;
                int32_t from;
                int32_t to;

                fp_t sum;

                num_batches = input->num_batches();
                num_channels = input->num_channels();
                height = input->height();
                width = input->width();

                for (int32_t batch = 0; batch < num_batches; batch++) {
                    for (int32_t channel = 0; channel < num_channels; channel++) {
                        from = MAX(0, channel - (n_ / 2));
                        to = MIN(num_channels - 1, channel + (n_ / 2));

                        for (int32_t row = 0; row < height; row++) {
                            for (int32_t column = 0; column < width; column++) {

                                sum = 0.0;

                                for (i = from; i <= to; i++) {
                                    sum += powf(input->access(batch, i, row, column, num_channels, height, width), 2);
                                }

                                output->access(batch, channel, row, column, num_channels, height, width) =
                                        input->access(batch, channel, row, column, num_channels, height, width) /
                                        powf((1 + (alpha_ / (fp_t) n_) * sum), beta_);
                            }
                        }
                    }
                }
            } else {
                PRINT_ERROR_AND_DIE("LRN not yet implemented for Tensor with num_dims != 4")
            }
        }
    }
}