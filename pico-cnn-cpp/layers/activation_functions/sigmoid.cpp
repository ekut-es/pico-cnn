//
// Created by junga on 03.04.20.
//

#include "sigmoid.h"
namespace pico_cnn {
    namespace naive {

        Sigmoid::Sigmoid(std::string name, uint32_t id, op_type op) : ActivationFunction(name, id, op) {

        }

        void Sigmoid::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void Sigmoid::activate(Tensor *input, Tensor *output) {
            uint32_t num_elements = input->num_elements();
            for(uint32_t element = 0; element < num_elements; element++) {
                output->access_blob(element) = 1 / (1 + expf(-input->access_blob(element)));

                // alternative formula:
                //  output->access_blob(element) = 0.5 * (1 + tanhf(input->access_blob(element) / 2));
            }
        }
    }
}