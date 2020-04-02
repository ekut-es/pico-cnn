/**
 * @brief Abstract base class for all operations
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_LAYER_H
#define PICO_CNN_LAYER_H

#include <string>

#include "../parameters.h"
#include "../tensor.h"

namespace pico_cnn {
    enum class op_type {
        Conv,
        Gemm,
        MaxPool,
        AveragePool,
        GlobalMaxPool,
        GlobalAveragePool,
        Relu,
        Softmax,
        LRN,
        BatchNormalization,
        Clip,
        MatMul,
        Mul,
        Add,
        Pad,
        Transpose,
        Concat,
        Reshape,
        Flatten,
        Squeeze,
        Unknown
    };
}

namespace pico_cnn {
    namespace naive {

        class Layer {
        public:
            Layer(std::string name, uint32_t id, op_type op);

            ~Layer();

            virtual void run(Tensor *input, Tensor *output) = 0;

            std::string name();

            uint32_t id();

            op_type op();

        private:
            std::string name_;
            uint32_t id_;
            op_type op_;
        };
    }
}

#endif //PICO_CNN_LAYER_H
