//
// Created by junga on 03.04.20.
//

#ifndef PICO_CNN_CLIP_H
#define PICO_CNN_CLIP_H

#include "pico-cnn-cpp/parameters.h"
#include "pico-cnn-cpp/tensor.h"
#include "pico-cnn-cpp/layers/layer.h"

#include "activation_function.h"

namespace pico_cnn {
    namespace naive {
        class Clip : ActivationFunction {
        public:
            Clip(std::string name, uint32_t id, op_type op, const fp_t min, const fp_t max);
            ~Clip();

            void run(Tensor *input, Tensor *output) override;

        private:
            const fp_t min, max;
            void activate(Tensor *input, Tensor *output) override;
        };
    }
}


#endif //PICO_CNN_CLIP_H
