#include "global_average_pooling.h"

pico_cnn::naive::GlobalAveragePooling::GlobalAveragePooling(std::string name, uint32_t id, pico_cnn::op_type op,
                                                            uint32_t *kernel_size, uint32_t *stride, uint32_t *padding):
                                                            Pooling(name, id, op, kernel_size, stride, padding) {

}

void pico_cnn::naive::GlobalAveragePooling::pool(pico_cnn::naive::Tensor *input, pico_cnn::naive::Tensor *output) {
    uint32_t num_dims = input->num_dimensions();

    uint32_t num_batches = input->num_batches();
    uint32_t num_channels = input->num_channels();
    uint32_t height = input->height();
    uint32_t width = input->width();
    uint32_t output_height = output->height();
    uint32_t output_width = output->width();

    fp_t global_sum = 0.0;

    if (num_dims == 4) {

        for (uint32_t batch = 0; batch < num_batches; batch++) {
            for (uint32_t channel = 0; channel < num_channels; channel++) {

                for (uint32_t row = 0; row < height; row++) {
                    for (uint32_t col = 0; col < width; col++) {

                        global_sum += input->access(batch, channel, row, col, num_channels, height, width);
                    }
                }
                output->access(batch, channel, 0, 0, num_channels, output_height, output_width) = global_sum / (fp_t)(height*width);

                global_sum = 0.0;
            }
        }

    } else if (num_dims == 3) {

        for (uint32_t batch = 0; batch < num_batches; batch++) {
            for (uint32_t channel = 0; channel < num_channels; channel++) {

                for (uint32_t row = 0; row < height; row++) {
                    for (uint32_t col = 0; col < width; col++) {

                        global_sum += input->access(batch, channel, col, num_channels, width);
                    }
                }
                output->access(batch, channel, 0, num_channels, output_width) = global_sum / (fp_t)(height*width);

                global_sum = 0.0;
            }
        }

    }
}
