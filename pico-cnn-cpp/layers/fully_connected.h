/**
 * @brief pico_cnn::naive::FullyConnected class provides naive implementation of FC operation
 * This implementation assumes the following data layout:
 * input: (1, X), kernel: (Y, X), bias: (1, Y), output: (1, Y)
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
            /**
             *
             * @param name
             * @param id
             * @param op
             * @param kernel We use the same data layout as used in the onnx file format: kernel->shape == (Y, X)
             * @param bias We use the same data layout as used in the onnx file format: bias->shape == (1, Y)
             */
            FullyConnected(std::string name, uint32_t id, op_type op, Tensor *kernel, Tensor *bias);
            ~FullyConnected() override = default;

            /**
             *
             * @param input input->shape == (1, X)
             * @param output output->shape == (1, Y)
             */
            void run(Tensor *input, Tensor *output) override;

        private:
            void gemm(Tensor *input, Tensor *output);

            Tensor *kernel_;
            Tensor *bias_;
        };

        class MatMul : Layer {
        public:
            MatMul(std::string name, uint32_t id, op_type op, Tensor *weights);
            ~MatMul() override = default;

            void run(Tensor *input, Tensor *output);

        private:
            void matmul(Tensor *input, Tensor *output);

            Tensor *weights_;
        };
    }
}

#endif //PICO_CNN_FULLY_CONNECTED_H
