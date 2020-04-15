/**
 * @brief pico_cnn::naive::ReLU provides the ReLU activation function
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_RE_LU_H
#define PICO_CNN_RE_LU_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

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

        class LeakyReLU : ActivationFunction {
        public:
            LeakyReLU(std::string name, uint32_t id, op_type op, fp_t leak);
            ~LeakyReLU() = default;

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;

            const fp_t leak_;
        };

        class ParameterizedReLU : ActivationFunction {
        public:
            ParameterizedReLU(std::string name, uint32_t id, op_type op, Tensor *slope);
            ~ParameterizedReLU() = default;

            void run(Tensor *input, Tensor *output) override;

        private:
            void activate(Tensor *input, Tensor *output) override;

            Tensor *slope_;
        };
    }
}


#endif //PICO_CNN_RE_LU_H
