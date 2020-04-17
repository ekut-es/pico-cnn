//
// Created by junga on 17.04.20.
//

#ifndef PICO_CNN_MAX_POOLING_H
#define PICO_CNN_MAX_POOLING_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "pooling.h"

namespace pico_cnn {
    namespace naive {
        class MaxPooling : public Pooling {
        public:
            MaxPooling(std::string name, uint32_t id, op_type op, uint32_t kernel_size, uint32_t stride, uint32_t *padding);
            ~MaxPooling() = default;

        private:
            void pool(Tensor *input, Tensor *output) override;
        };
    }
}
#endif //PICO_CNN_MAX_POOLING_H
