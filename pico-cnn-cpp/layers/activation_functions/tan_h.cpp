//
// Created by junga on 03.04.20.
//

#include "tan_h.h"
namespace pico_cnn {
    namespace naive {

        TanH::TanH(std::string name, uint32_t id, op_type op) : ActivationFunction(name, id, op) {

        }

        void TanH::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void TanH::activate(Tensor *input, Tensor *output) {
            uint32_t num_elements = input->num_elements();

            for(uint32_t i = 0; i < num_elements; i++) {
                output->access_blob(i) = tanhf(input->access_blob(i));
            }
        }
    }
}