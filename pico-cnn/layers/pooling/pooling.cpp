#include "pooling.h"

pico_cnn::naive::Pooling::Pooling(std::string name, uint32_t id, pico_cnn::op_type op, uint32_t *kernel_size,
                                  uint32_t *stride, uint32_t *padding) : Layer(name, id, op) {

    if (kernel_size) {
        kernel_size_ = new uint32_t[2]();
        std::memcpy(kernel_size_, kernel_size, 2 * sizeof(uint32_t));
    } else {
        kernel_size_ = kernel_size;
    }

    if (stride) {
        stride_ = new uint32_t[2]();
        std::memcpy(stride_, stride, 2 * sizeof(uint32_t));
    } else {
        stride_ = stride;
    }

    if (padding) {
        padding_ = new uint32_t[4]();
        std::memcpy(padding_, padding, 4 * sizeof(uint32_t));
    } else {
        padding_ = padding;
    }
}

pico_cnn::naive::Pooling::~Pooling() {
    delete [] kernel_size_;
    delete [] stride_;
    delete [] padding_;
}

void pico_cnn::naive::Pooling::run(pico_cnn::naive::Tensor *input, pico_cnn::naive::Tensor *output) {

    if (input->num_dimensions() == 4 || input->num_dimensions() == 3) {

        Tensor *input_tensor;

        if (padding_) {
            input_tensor = input->expand_with_padding(padding_);
        } else {
            input_tensor = input;
        }

        this->pool(input_tensor, output);

        if (padding_) {
            delete input_tensor;
        }
    } else {
        PRINT_ERROR_AND_DIE("Not implemented for Tensor with num_dims: " << input->num_dimensions());
    }
}