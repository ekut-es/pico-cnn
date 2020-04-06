#include "fully_connected.h"

namespace pico_cnn {
    namespace naive {

        FullyConnected::FullyConnected(std::string name, uint32_t id, op_type op, Tensor *kernel, Tensor *bias) :
        Layer(name, id, op) {
            kernel_ = kernel;
            bias_ = bias;
        }

        void FullyConnected::run(Tensor *input, Tensor *output) {
            // TODO: Implement checks for dimensions
            this->gemm(input, output);
        }

        void FullyConnected::gemm(Tensor *input, Tensor *output) {

            uint32_t output_width = output->shape()->operator[](1);
            uint32_t input_width = input->shape()->operator[](1);

            for (uint32_t i = 0; i < output_width; i++) {

                fp_t pixel = 0.0;

                for (uint32_t j = 0; j < input_width; j++) {
                    pixel += input->access(0, j) * kernel_->access(j, i);
                }

                if(bias_)
                    pixel += bias_->access(0, i);

                output->access(0, i) = pixel;
            }
        }
    }
}