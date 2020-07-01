/**
 * @brief Clip activation function, clipping all values between the given minimum and maximum.
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_CLIP_H
#define PICO_CNN_CLIP_H

#include "../../parameters.h"
#include "../../tensor.h"
#include "../layer.h"

#include "activation_function.h"

namespace pico_cnn {
    namespace naive {
        class Clip : ActivationFunction {
        public:
            Clip(std::string name, uint32_t id, op_type op, const fp_t min, const fp_t max);
            ~Clip();

            void run(Tensor *input, Tensor *output) override;

        private:
            const fp_t min, max;
            void activate(Tensor *input, Tensor *output) override;
        };
    }
}


#endif //PICO_CNN_CLIP_H
