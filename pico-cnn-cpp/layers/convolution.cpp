#include "convolution.h"

namespace pico_cnn {
    namespace naive {

        Convolution::Convolution(std::string name, uint32_t id, op_type op, Tensor *kernel, Tensor *bias,
                uint32_t *padding, uint32_t *stride, uint32_t num_groups) : Layer(name, id, op) {

            kernel_ = kernel;
            bias_ = bias;
            padding_ = padding;
            stride_ = stride;
            num_groups_ = num_groups;
        }

        void Convolution::run(Tensor *input, Tensor *output) {
            // TODO: Maybe use different methods to convolve? conv1d, conv2d?
            // TODO: Check dimensions?

            uint32_t num_input_channels = input->num_channels();
            uint32_t input_height = input->height();
            uint32_t input_width = input->width();

            uint32_t num_output_channels = output->num_channels();
            uint32_t output_height = output->height();
            uint32_t output_width = output->width();

            auto *tmp_shape = new TensorShape(output->num_batches(), num_output_channels, output_height, output_width);
            auto *tmp_tensor = new Tensor(tmp_shape);


            for (uint32_t g = 0; g < num_groups_; g++) {
                for (uint32_t i = g*num_output_channels/num_groups_; i < (g+1)*num_output_channels/num_groups_; i++) {

                    this->convolve(input, output, g*num_input_channels/num_groups_, i, g);

                    if (num_input_channels > num_groups_) {
                        uint32_t cnt = 1;

                        for (uint32_t j = g*num_input_channels/num_groups_+1; j < (g+1)*(num_input_channels/num_groups_); j++) {

                            this->convolve(input, tmp_tensor, j, i, g);

                            output->add_tensor(tmp_tensor);

                            cnt++;
                        }
                    }
                }
            }

            delete tmp_tensor;
            delete tmp_shape;
        }

        void Convolution::convolve(Tensor *input, Tensor *output, uint32_t input_channel, uint32_t output_channel, uint32_t group_idx) {

            uint32_t num_input_channels = input->num_channels();
            uint32_t input_height = input->height();
            uint32_t input_width = input->width();

            uint32_t kernel_height = kernel_->height();
            uint32_t kernel_width = kernel_->width();

            uint32_t channel_row, channel_col;
            uint32_t kernel_row, kernel_col;
            uint32_t height_crop = kernel_height/2;
            uint32_t width_crop = kernel_width/2;

            uint32_t stride_height = stride_[0];
            uint32_t stride_width = stride_[1];

            fp_t pixel;

            uint32_t output_channel_row = 0;
            uint32_t output_channel_col = 0;

            uint32_t output_channel_width = (input_width - kernel_width)/stride_width + 1;

            for(channel_row = height_crop; channel_row < input_height-height_crop; channel_row+=stride_height) {
                for(channel_col = width_crop; channel_col < input_width - width_crop; channel_col += stride_width) {
                    pixel = 0.0;

                    for(kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
                        for(kernel_col = 0; kernel_col < kernel_width; kernel_col++) {
//                            pixel += kernel[kernel_row*kernel_width + kernel_col] *
//                                     input_channel[input_width*(channel_row-height_crop+kernel_row) + channel_col-width_crop+kernel_col];
                            pixel += kernel_->access(output_channel, input_channel, kernel_row, kernel_col) *
                                    input->access(0, group_idx*num_input_channels/num_groups_,
                                            channel_row-height_crop+kernel_row, channel_col-width_crop+kernel_col);
                        }
                    }

                    if (bias_)
                        pixel += bias_->access(output_channel);

                    //output_channel[output_channel_row*output_channel_width+output_channel_col] = pixel;
                    output->access(0, output_channel, output_channel_row, output_channel_col) = pixel;
                    output_channel_col++;

                }
                output_channel_row++;
                output_channel_col = 0;

            }
        }
    }
}