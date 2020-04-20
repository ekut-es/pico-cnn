/**
 * @brief pico_cnn::naive::Pooling class is an abstract class serving as the base class from all pooling operations.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_POOLING_H
#define PICO_CNN_POOLING_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

namespace pico_cnn {
    namespace naive {
        class Pooling : Layer {
        public:
            Pooling(std::string name, uint32_t id, op_type op, uint32_t *kernel_size, uint32_t *stride, uint32_t *padding);
            ~Pooling() = default;

            void run(Tensor *input, Tensor *output) override;

        protected:
            virtual void pool(Tensor *input, Tensor *output) = 0;

            uint32_t *kernel_size_;
            uint32_t *stride_;
            uint32_t *padding_;
        };
    }
}

#endif //PICO_CNN_POOLING_H
