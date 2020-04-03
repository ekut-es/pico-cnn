//
// Created by junga on 03.04.20.
//

#ifndef PICO_CNN_LRN_H
#define PICO_CNN_LRN_H

#include <cmath>

#include "pico-cnn-cpp/parameters.h"
#include "pico-cnn-cpp/tensor.h"
#include "pico-cnn-cpp/layers/layer.h"

#include "activation_function.h"

namespace pico_cnn {
    namespace naive {
        class LRN : ActivationFunction {
        public:
            LRN(std::string name, uint32_t id, op_type op, const fp_t alpha, const fp_t beta, const uint16_t n);
            ~LRN();

            void run(Tensor *input, Tensor *output) override;

        private:
            const fp_t alpha_, beta_;
            const uint16_t n_;

            void activate(Tensor *input, Tensor *output) override;
        };
    }
}

#endif //PICO_CNN_LRN_H
