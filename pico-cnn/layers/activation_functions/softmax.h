/**
 * @brief Softmax activation function.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_SOFTMAX_H
#define PICO_CNN_SOFTMAX_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "activation_function.h"

#include <cmath>

namespace pico_cnn {
    namespace naive {
        class Softmax : ActivationFunction {
        public:
            Softmax(std::string name, uint32_t id, op_type op);
            ~Softmax();

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;
        };
    }
}

#endif //PICO_CNN_SOFTMAX_H
