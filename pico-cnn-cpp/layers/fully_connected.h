/**
 * @brief pico_cnn::naive::FullyConnected class provides naive implementation of FC operation
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_FULLY_CONNECTED_H
#define PICO_CNN_FULLY_CONNECTED_H

#include "../parameters.h"
#include "../tensor.h"
#include "layer.h"

namespace pico_cnn {
    namespace naive {
        class FullyConnected : Layer {
        public:
            FullyConnected(std::string name, uint32_t id, op_type op, Tensor *kernel, Tensor *bias);
            ~FullyConnected() = default;

            void run(Tensor *input, Tensor *output) override;

        private:
            void gemm(Tensor *input, Tensor *output);

            Tensor *kernel_;
            Tensor *bias_;
        };
    }
}

#endif //PICO_CNN_FULLY_CONNECTED_H
