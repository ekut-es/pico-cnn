//
// Created by junga on 03.04.20.
//

#ifndef PICO_CNN_TAN_H_H
#define PICO_CNN_TAN_H_H

#include "pico-cnn-cpp/parameters.h"
#include "pico-cnn-cpp/tensor.h"
#include "pico-cnn-cpp/layers/layer.h"

#include "activation_function.h"

namespace pico_cnn {
    namespace naive {
        class TanH : ActivationFunction {
        public:
            TanH(std::string name, uint32_t id, op_type op);
            ~TanH();

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;
        };
    }
}

#endif //PICO_CNN_TAN_H_H
