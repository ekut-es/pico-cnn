#include "pooling.h"

pico_cnn::naive::Pooling::Pooling(std::string name, uint32_t id, pico_cnn::op_type op, uint32_t kernel_size,
                                  uint32_t stride, uint32_t *padding) : Layer(name, id, op) {

    kernel_size_ = kernel_size;
    stride_ = stride;
    padding_ = padding;
}

void pico_cnn::naive::Pooling::run(pico_cnn::naive::Tensor *input, pico_cnn::naive::Tensor *output) {
    if (input->shape()->num_dimensions() != 4) {
        PRINT_ERROR_AND_DIE("Not implemented for TensorShape: " << *input->shape());
    }

    uint32_t num_input_channels = input->num_channels();
    uint32_t input_height = input->height();
    uint32_t input_width = input->width();

    uint32_t num_output_channels = output->num_channels();
    uint32_t output_height = output->height();
    uint32_t output_width = output->width();

    Tensor *input_tensor;

    if (padding_) {
        input_tensor = input->expand_with_padding(padding_);
    } else {
        input_tensor = input;
    }

    this->pool(input_tensor, output);

    if (padding_) {
        delete input_tensor->shape();
        delete input_tensor;
    }
}