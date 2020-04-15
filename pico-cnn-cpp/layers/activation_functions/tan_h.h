//
// Created by junga on 03.04.20.
//

#ifndef PICO_CNN_TAN_H_H
#define PICO_CNN_TAN_H_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "activation_function.h"

#include <cmath>

namespace pico_cnn {
    namespace naive {
        class TanH : ActivationFunction {
        public:
            TanH(std::string name, uint32_t id, op_type op);
            ~TanH() = default;

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;
        };
    }
}

#endif //PICO_CNN_TAN_H_H
