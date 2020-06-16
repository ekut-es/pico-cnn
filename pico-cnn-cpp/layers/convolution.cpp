#include "convolution.h"

namespace pico_cnn {
    namespace naive {

        Convolution::Convolution(std::string name, uint32_t id, op_type op, Tensor *kernel, Tensor *bias,
                                 uint32_t *padding, uint32_t *stride, uint32_t num_groups) : Layer(name, id, op) {

            kernel_ = kernel;
            bias_ = bias;

            if (padding) {
                padding_ = new uint32_t[4]();
                std::memcpy(padding_, padding, 4 * sizeof(uint32_t));
            } else {
                padding_ = padding;
            }

            stride_ = new uint32_t[2]();
            std::memcpy(stride_, stride, 2*sizeof(uint32_t));

            num_groups_ = num_groups;

            kernel_height = kernel_->height();
            kernel_width = kernel_->width();
        }

        Convolution::~Convolution() {
            delete [] padding_;
            delete [] stride_;
        }

        void Convolution::run(Tensor *input, Tensor *output) {

            if (input->num_dimensions() != 4 && input->num_dimensions() != 3) {
                PRINT_ERROR("Not implemented for Tensor with number of dimensions: " << input->num_dimensions());
            }

            uint32_t num_batches = input->num_batches();
            if (num_batches != 1)
                PRINT_ERROR_AND_DIE("Number of batches != 1. Convolution not tested for this.")

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

            auto *tmp_tensor = new Tensor(num_batches, num_output_channels, output_height, output_width);

            for (uint32_t batch = 0; batch < num_batches; batch++) {

                for (uint32_t g = 0; g < num_groups_; g++) {
                    for (uint32_t i = g * num_output_channels / num_groups_;
                         i < (g + 1) * num_output_channels / num_groups_; i++) {

                        this->convolve(input_tensor, output, g * num_input_channels / num_groups_, i, 0, input_height, input_width, kernel_height, kernel_width, stride_, num_input_channels, num_output_channels, output_height, output_width);

                        if (num_input_channels > num_groups_) {
                            uint32_t cnt = 1;

                            for (uint32_t j = g * num_input_channels / num_groups_ + 1;
                                 j < (g + 1) * (num_input_channels / num_groups_); j++) {

                                this->convolve(input_tensor, tmp_tensor, j, i, cnt, input_height, input_width, kernel_height, kernel_width, stride_, num_input_channels, num_output_channels, output_height, output_width);

//                                output->add_channel(tmp_tensor, 0, i);
                                output->add_tensor(tmp_tensor);
//                                for (uint32_t tmp = 0; tmp < output_height; tmp++){
//                                    for (uint32_t tmp2 = 0; tmp2 < output_width; tmp2++) {
//                                        output->data_[(0*num_output_channels*output_height*output_width) + (i*output_height*output_width) + (tmp*output_width) + tmp2] +=
//                                                tmp_tensor->data_[(0*num_output_channels*output_height*output_width) + (i*output_height*output_width) + (tmp*output_width) + tmp2];
//                                    }
//                                }

                                cnt++;
                            }
                        }
                    }
                }
            }

            delete tmp_tensor;

            if (padding_) {
                delete input_tensor;
            }


        }

        void Convolution::convolve(Tensor *input, Tensor *output, uint32_t input_channel, uint32_t output_channel,
                                   uint32_t cnt, uint32_t input_height, uint32_t input_width,
                                   uint32_t kernel_height, uint32_t kernel_width, uint32_t *stride_,
                                   uint32_t num_input_channels, uint32_t num_output_channels,
                                   uint32_t output_height, uint32_t output_width) {

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

                            pixel += kernel_->access(output_channel, cnt, kernel_row, kernel_col,
                                                     num_input_channels, kernel_height, kernel_width) *
                                     input->access(0, input_channel, channel_row-height_crop+kernel_row, channel_col-width_crop+kernel_col,
                                                   num_input_channels, input_height, input_width);
//                            pixel += kernel_->data_[(output_channel*num_input_channels*kernel_height*kernel_width) + (cnt*kernel_height*kernel_width) + (kernel_row*kernel_width) + kernel_col] *
//                                     input->data_[(0*num_input_channels*input_height*input_width) + (input_channel*input_height*input_width) + ((channel_row-height_crop+kernel_row)*input_width) + (channel_col-width_crop+kernel_col)];
                        }
                    }

                    if (input_channel == 0 && bias_) {
//                        pixel += bias_->data_[output_channel];
                        pixel += bias_->access(output_channel);
                    }

                    output->access(0, output_channel, output_channel_row, output_channel_col,
                                   num_output_channels, output_height, output_width) = pixel;
//                    output->data_[(0*num_output_channels*output_height*output_width) + (output_channel*output_height*output_width) + (output_channel_row*output_width) + output_channel_col] = pixel;
                    output_channel_col++;

                }
                output_channel_row++;
                output_channel_col = 0;

            }
        }

//        void Convolution::run(Tensor *input, Tensor *output) {
//            printf("Why here?\n");
//        }
    }
}