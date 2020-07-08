#include "activation_function.h"

namespace pico_cnn {
    namespace naive {

        ActivationFunction::ActivationFunction(std::string name, uint32_t id, op_type op) : Layer(name, id, op) {

        }

        ActivationFunction::~ActivationFunction() = default;

        void ActivationFunction::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void ActivationFunction::activate(Tensor *input, Tensor *output) {
            input->copy_data_into(output);
        }
    }
}