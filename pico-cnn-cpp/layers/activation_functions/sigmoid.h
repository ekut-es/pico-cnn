//
// Created by junga on 03.04.20.
//

#ifndef PICO_CNN_SIGMOID_H
#define PICO_CNN_SIGMOID_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "activation_function.h"

namespace pico_cnn {
    namespace naive {
        class Sigmoid : ActivationFunction {
        public:
            Sigmoid(std::string name, uint32_t id, op_type op);
            ~Sigmoid() = default;

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;
        };
    }
}

#endif //PICO_CNN_SIGMOID_H
