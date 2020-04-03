//
// Created by junga on 03.04.20.
//

#ifndef PICO_CNN_SOFTMAX_H
#define PICO_CNN_SOFTMAX_H

#include "pico-cnn-cpp/parameters.h"
#include "pico-cnn-cpp/tensor.h"
#include "pico-cnn-cpp/layers/layer.h"

#include "activation_function.h"

namespace pico_cnn {
    namespace naive {
        class Softmax : ActivationFunction {
        public:
            Softmax(std::string name, uint32_t id, op_type op);
            ~Softmax();

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;
        };
    }
}

#endif //PICO_CNN_SOFTMAX_H
