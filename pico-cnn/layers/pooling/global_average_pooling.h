/**
 * @brief Global Average Pooling operation.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_GLOBAL_AVERAGE_POOLING_H
#define PICO_CNN_GLOBAL_AVERAGE_POOLING_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "pooling.h"

namespace pico_cnn {
    namespace naive {
        class GlobalAveragePooling : public Pooling {
        public:
            GlobalAveragePooling(std::string name, uint32_t id, op_type op, uint32_t *kernel_size = nullptr,
                    uint32_t *stride = nullptr, uint32_t *padding = nullptr);
            ~GlobalAveragePooling() = default;

        private:
            void pool(Tensor *input, Tensor *output) override;
        };
    }
}

#endif //PICO_CNN_GLOBAL_AVERAGE_POOLING_H
