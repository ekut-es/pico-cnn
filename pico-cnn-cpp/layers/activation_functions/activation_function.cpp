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
            if(input->copy_data_into(output) != 0)
                PRINT_ERROR_AND_DIE("Attempted to copy Tensors of unequal shapes.");
        }
    }
}