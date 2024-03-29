/**
 * @brief Average Pooling operation.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_AVERAGE_POOLING_H
#define PICO_CNN_AVERAGE_POOLING_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "pooling.h"

namespace pico_cnn {
    namespace naive {
        class AveragePooling : public Pooling {
        public:
            AveragePooling(std::string name, uint32_t id, op_type op, uint32_t *kernel_size, uint32_t *stride, uint32_t *padding, bool count_include_pad);
            ~AveragePooling() = default;

        private:
            void pool(Tensor *input, Tensor *output) override;

            bool count_include_pad_;
        };
    }
}

#endif //PICO_CNN_AVERAGE_POOLING_H
