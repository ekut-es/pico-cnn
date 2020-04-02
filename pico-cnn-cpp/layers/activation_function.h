/**
 * @brief pico_cnn::naive::ActivationFunction serves as base class for all activation functions.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_ACTIVATION_FUNCTIONS_H
#define PICO_CNN_ACTIVATION_FUNCTIONS_H

#include "../parameters.h"
#include "../tensor.h"
#include "layer.h"

namespace pico_cnn {
    namespace naive {
        class ActivationFunction : Layer {
        public:
            ActivationFunction(std::string name, uint32_t id, op_type op);

            ~ActivationFunction();

            virtual void run(Tensor *input, Tensor *output);

        protected:
            /**
             * Identity serving as the most basic activation function
             * @param data
             * @return the tensor that was given
             */
            virtual void activate(Tensor *input, Tensor *output);

        };

        class ReLU : ActivationFunction {
        public:
            ReLU(std::string name, uint32_t id, op_type op);
            ~ReLU();

            void run(Tensor *input, Tensor *output);

        private:
            void activate(Tensor *input, Tensor *output);

        };
    }
}

#endif //PICO_CNN_ACTIVATION_FUNCTIONS_H
