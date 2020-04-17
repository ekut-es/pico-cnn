#include "max_pooling.h"

pico_cnn::naive::MaxPooling::MaxPooling(std::string name, uint32_t id, pico_cnn::op_type op, uint32_t kernel_size,
                                        uint32_t stride, uint32_t *padding) :
                                        Pooling(name, id, op, kernel_size, stride, padding) {

}

void pico_cnn::naive::MaxPooling::pool(pico_cnn::naive::Tensor *input, pico_cnn::naive::Tensor *output) {
    uint32_t num_batches = input->num_batches();
    uint32_t num_channels = input->num_channels();
    uint32_t height = input->height();
    uint32_t width = input->width();

    uint32_t output_channel_row, output_channel_column;
    uint32_t output_channel_height, output_channel_width;


    output_channel_height = (height-kernel_size_)/stride_+1;
    output_channel_width = (width-kernel_size_)/stride_+1;

    fp_t pixel, candidate;

    for (uint32_t batch = 0; batch < num_batches; batch++) {
        for (uint32_t channel = 0; channel < num_channels; channel++) {

            output_channel_row = 0;
            output_channel_column = 0;

            for (uint32_t channel_row = 0; channel_row < height && output_channel_row < output_channel_height; channel_row += stride_) {
                for (uint32_t channel_column = 0; channel_column < width && output_channel_column < output_channel_width; channel_column += stride_) {

                    pixel = input->access(batch, channel, channel_row, channel_column);

                    for (uint32_t kernel_row = channel_row; kernel_row < channel_row + kernel_size_ && kernel_row < height; kernel_row++) {
                        for (uint32_t kernel_column = channel_column; kernel_column < channel_column + kernel_size_ && kernel_column < width; kernel_column++) {

                            candidate = input->access(batch, channel, kernel_row, kernel_column);
                            if (candidate > pixel) {
                                pixel = candidate;
                            }
                        }
                    }

                    output->access(batch, channel, output_channel_row, output_channel_column) = pixel;
                    output_channel_column++;
                }
                output_channel_row++;
                output_channel_column = 0;
            }
        }
    }
}
