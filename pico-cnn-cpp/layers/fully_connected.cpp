#include "fully_connected.h"

namespace pico_cnn {
    namespace naive {

        FullyConnected::FullyConnected(std::string name, uint32_t id, op_type op, Tensor *kernel, Tensor *bias) :
        Layer(name, id, op) {
            kernel_ = kernel;
            bias_ = bias;
        }

        void FullyConnected::run(Tensor *input, Tensor *output) {
            if (input->num_dimensions() != 2 || output->num_dimensions() != 2) {
                PRINT_ERROR_AND_DIE("Fully connected operation only supports 2D input and output.")
            }
            this->gemm(input, output);
        }

        void FullyConnected::gemm(Tensor *input, Tensor *output) {

            uint32_t output_width = output->width();
            uint32_t input_width = input->width();
            uint32_t kernel_width = kernel_->width();

            for (uint32_t i = 0; i < output_width; i++) {

                fp_t pixel = 0.0;

                for (uint32_t j = 0; j < input_width; j++) {
                    // With the kernel layout used in the onnx format we can use this memory optimized
                    // access pattern (i, j) instead of (j, i) as the "normal" matrix multiplication is defined
                    pixel += input->access(0, j, input_width) * kernel_->access(i, j, kernel_width);
                }

                if(bias_) {
                    pixel += bias_->access(i);
                }

                output->access(0, i, output_width) = pixel;
            }
        }
    }
}