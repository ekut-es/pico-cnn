/**
 * @brief pico_cnn::naive::Convolution provides naive implementation of convolution operation
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_CONVOLUTION_H
#define PICO_CNN_CONVOLUTION_H

#include "../parameters.h"
#include "../tensor.h"
#include "layer.h"

namespace pico_cnn {
    namespace naive {
        class Convolution : Layer {
        public:
            Convolution(std::string name, uint32_t id, op_type op, Tensor *kernel, Tensor *bias,
                        uint32_t *padding, uint32_t *stride, uint32_t num_groups);
            ~Convolution() = default;

            void run(Tensor *input, Tensor *output) override;

        private:
            void convolve(Tensor *input, Tensor *output, uint32_t input_channel, uint32_t output_channel, uint32_t cnt);

            Tensor *kernel_;
            Tensor *bias_;
            uint32_t *padding_;
            uint32_t *stride_;
            uint32_t num_groups_;
        };
    }
}

#endif //PICO_CNN_CONVOLUTION_H
