#include "average_pooling.h"

pico_cnn::naive::AveragePooling::AveragePooling(std::string name, uint32_t id, pico_cnn::op_type op,
                                                uint32_t *kernel_size, uint32_t *stride, uint32_t *padding,
                                                bool count_include_pad) :
                                                Pooling(name, id, op, kernel_size, stride, padding) {

    count_include_pad_ = count_include_pad;
}

void pico_cnn::naive::AveragePooling::pool(pico_cnn::naive::Tensor *input, pico_cnn::naive::Tensor *output) {
    uint32_t num_dims = input->shape()->num_dimensions();

    uint32_t num_batches = input->num_batches();
    uint32_t num_channels = input->num_channels();
    uint32_t height = input->height();
    uint32_t width = input->width();

    uint32_t channel_row, channel_column;
    uint32_t output_channel_row, output_channel_column;
    uint32_t output_channel_height, output_channel_width;

    uint32_t kernel_row, kernel_column;

    output_channel_height = (height-kernel_size_[0])/stride_[0]+1;
    output_channel_width = (width-kernel_size_[1])/stride_[1]+1;

    if(count_include_pad_ == 1) {

        for (uint32_t batch = 0; batch < num_batches; batch++) {
            for (uint32_t channel = 0; channel < num_channels; channel++) {

                output_channel_row = 0;
                output_channel_column = 0;

                for (channel_row = 0;
                     channel_row < height && output_channel_row < output_channel_height; channel_row += stride_[0]) {
                    for (channel_column = 0;
                         channel_column < width &&
                         output_channel_column < output_channel_width; channel_column += stride_[1]) {
                        fp_t pixel = 0.0;

                        for (kernel_row = channel_row;
                             kernel_row < channel_row + kernel_size_[0] && kernel_row < height; kernel_row++) {
                            for (kernel_column = channel_column;
                                 kernel_column < channel_column + kernel_size_[1] &&
                                 kernel_column < width; kernel_column++) {
                                if (num_dims == 4) {
                                    pixel += input->access(batch, channel, kernel_row, kernel_column);
                                } else if (num_dims == 3) {
                                    pixel += input->access(batch, channel, kernel_column);
                                }
                            }
                        }

                        if (num_dims == 4) {
                            output->access(batch, channel, output_channel_row, output_channel_column) = pixel /
                                    ((fp_t) (kernel_size_[0] * kernel_size_[1]));
                        } else if (num_dims == 3) {
                            output->access(batch, channel, output_channel_column) = pixel /
                                    ((fp_t) (kernel_size_[0] * kernel_size_[1]));
                        }
                        output_channel_column++;
                    }
                    output_channel_row++;
                    output_channel_column = 0;
                }
            }
        }

    } else if(count_include_pad_ == 0) {

        uint32_t crop[2] = {kernel_size_[0]/2, kernel_size_[1]/2};

        for (uint32_t batch = 0; batch < num_batches; batch++) {
            for (uint32_t channel = 0; channel < num_channels; channel++) {

                output_channel_row = 0;
                output_channel_column = 0;

                for (channel_row = 0;
                     channel_row < height && output_channel_row < output_channel_height; channel_row += stride_[0]) {
                    for (channel_column = 0;
                         channel_column < width &&
                         output_channel_column < output_channel_width; channel_column += stride_[1]) {
                        fp_t pixel = 0.0;

                        for (kernel_row = channel_row;
                             kernel_row < channel_row + kernel_size_[0] && kernel_row < height; kernel_row++) {
                            for (kernel_column = channel_column;
                                 kernel_column < channel_column + kernel_size_[1] &&
                                 kernel_column < width; kernel_column++) {
                                if (num_dims == 4) {
                                    pixel += input->access(batch, channel, kernel_row, kernel_column);
                                } else if (num_dims == 3) {
                                    pixel += input->access(batch, channel, kernel_column);
                                }
                            }
                        }

                        // center case
                        if (output_channel_row >= crop[0] && output_channel_row < output_channel_height - crop[0] &&
                            output_channel_column >= crop[1] && output_channel_column < output_channel_width - crop[1]) {

                            if (num_dims == 4) {
                                output->access(batch, channel, output_channel_row, output_channel_column) = pixel /
                                        ((fp_t) (kernel_size_[0] * kernel_size_[1]));
                            } else if (num_dims == 3) {
                                output->access(batch, channel, output_channel_column) = pixel /
                                        ((fp_t) (kernel_size_[0] * kernel_size_[1]));
                            }

                        // edge case
                        } else {

                            uint32_t up_row, down_row, left_col, right_col;
                            int32_t divisor, div_row, div_col;

                            up_row = MAX(channel_row, crop[0]);
                            down_row = MIN(channel_row + kernel_size_[0] - 1, height - crop[0] - 1);
                            div_row = down_row - up_row + 1;

                            left_col = MAX(channel_column, crop[1]);
                            right_col = MIN(channel_column + kernel_size_[1] - 1, width - crop[1] - 1);
                            div_col = right_col - left_col + 1;

                            divisor = div_row * div_col;

                            if (divisor == 0) {
                                PRINT_ERROR("ERROR: Division by zero! Aborting execution.")
                                exit(1);
                            }

                            if (num_dims == 4) {
                                output->access(batch, channel, output_channel_row, output_channel_column) = pixel /
                                        ((fp_t) (divisor));
                            } else if (num_dims == 3) {
                                output->access(batch, channel, output_channel_column) = pixel /
                                        ((fp_t) (divisor));
                            }

                        }


                        output_channel_column++;
                    }
                    output_channel_row++;
                    output_channel_column = 0;
                }
            }
        }
    } else {
        PRINT_ERROR("ERROR: Unsupported values for 'count_include_pad'.")
    }
}
