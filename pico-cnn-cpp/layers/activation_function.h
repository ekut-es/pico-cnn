/**
 * @brief pico_cnn::naive::ActivationFunction serves as base class for all activation functions.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_ACTIVATION_FUNCTIONS_H
#define PICO_CNN_ACTIVATION_FUNCTIONS_H

#include "../parameters.h"
#include "../tensor.h"
#include "layer.h"

namespace pico_cnn {
    namespace naive {
        class ActivationFunction : Layer {
        public:
            ActivationFunction(std::string name, uint32_t id, op_type op);

            ~ActivationFunction();

            Tensor run(Tensor data);

        protected:
            Tensor activate(Tensor data);

        };
    }
}

#endif //PICO_CNN_ACTIVATION_FUNCTIONS_H
