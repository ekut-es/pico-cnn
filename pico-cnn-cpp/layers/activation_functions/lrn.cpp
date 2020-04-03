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
            if(input->shape()->num_dimensions() != 3)
                PRINT_ERROR_AND_DIE("LRN can only be used with TensorShape: (num_channels, height, width) but got: " << input->shape())
            else
                this->activate(input, output);
        }

        void LRN::activate(Tensor *input, Tensor *output) {
            int32_t i, depth, height, width;
            int32_t from;
            int32_t to;

            fp_t sum;

            depth = input->shape()->operator[](0);
            height = input->shape()->operator[](1);
            width = input->shape()->operator[](2);

            for(int32_t channel = 0; channel < depth; channel++) {
                from = MAX(0,channel-(n_/2));
                to = MIN(depth-1,channel+(n_/2));

                for(int32_t row = 0; row < height; row++) {
                    for(int32_t column = 0; column < width; column++) {

                        sum = 0.0;

                        for(i = from; i <= to; i++) {
                            sum += powf(input->access(i, row, column), 2);
                        }

                        output->access(channel, row, column) = input->access(channel, row, column) /
                                powf((1+(alpha_/(fp_t)n_)*sum), beta_);
                    }
                }
            }
        }
    }
}