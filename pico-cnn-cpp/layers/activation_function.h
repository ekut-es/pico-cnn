//
// Created by junga on 17.03.20.
//

#ifndef PICO_CNN_ACTIVATION_FUNCTIONS_H
#define PICO_CNN_ACTIVATION_FUNCTIONS_H

#include "../parameters.h"
#include "../tensor.h"
#include "layer.h"

namespace pico_cnn {
    class ActivationFunction: Layer {
    public:
        ActivationFunction(std::string name, uint32_t id, op_type op);
        ~ActivationFunction();

        Tensor run(Tensor data);

    protected:
        Tensor activate(Tensor data);

    };
}

#endif //PICO_CNN_ACTIVATION_FUNCTIONS_H
