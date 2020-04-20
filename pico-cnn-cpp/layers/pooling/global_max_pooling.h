//
// Created by junga on 20.04.20.
//

#ifndef PICO_CNN_GLOBAL_MAX_POOLING_H
#define PICO_CNN_GLOBAL_MAX_POOLING_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "pooling.h"

namespace pico_cnn {
    namespace naive {
        class GlobalMaxPooling : public Pooling {
        public:
            GlobalMaxPooling(std::string name, uint32_t id, op_type op, uint32_t *kernel_size = nullptr,
                    uint32_t *stride = nullptr, uint32_t *padding = nullptr);
            ~GlobalMaxPooling() = default;

        private:
            void pool(Tensor *input, Tensor *output) override;

        };
    }
}

#endif //PICO_CNN_GLOBAL_MAX_POOLING_H
