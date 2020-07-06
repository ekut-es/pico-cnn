#include "softmax.h"
namespace pico_cnn {
    namespace naive {

        Softmax::Softmax(std::string name, uint32_t id, op_type op) : ActivationFunction(name, id, op) {

        }

        Softmax::~Softmax() {

        }

        void Softmax::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void Softmax::activate(Tensor *input, Tensor *output) {
            long double denominator = 0.0;

            uint32_t num_elements = input->num_elements();

            for(uint32_t i = 0; i < num_elements; i++) {
                denominator += expl((long double)input->access_blob(i));
            }

            for(uint32_t i = 0; i < num_elements; i++) {
                output->access_blob(i) = (fp_t)(expl((long double)input->access_blob(i)) / denominator);
            }
        }
    }
}