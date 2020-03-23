#include "activation_function.h"

namespace pico_cnn {
    namespace naive {

        ActivationFunction::ActivationFunction(std::string name, uint32_t id, op_type op) : Layer(name, id, op) {

        }

        ActivationFunction::~ActivationFunction() = default;

        Tensor ActivationFunction::run(Tensor data) {
            //return this->activate(data);
        }

        Tensor ActivationFunction::activate(Tensor data) {

        }
    }
}