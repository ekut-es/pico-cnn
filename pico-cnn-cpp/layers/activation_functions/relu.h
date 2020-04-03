/**
 * @brief pico_cnn::naive::ReLU provides the ReLU activation function
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_RE_LU_H
#define PICO_CNN_RE_LU_H

#include "pico-cnn-cpp/parameters.h"
#include "pico-cnn-cpp/tensor.h"
#include "pico-cnn-cpp/layers/layer.h"

#include "activation_function.h"

namespace pico_cnn {
    namespace naive {
        class ReLU : ActivationFunction {
        public:
            ReLU(std::string name, uint32_t id, op_type op);

            ~ReLU();

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;

        };
    }
}


#endif //PICO_CNN_RE_LU_H
