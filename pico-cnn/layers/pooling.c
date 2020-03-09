#include "pooling.h"

void max_pooling1d_naive(const fp_t* input_channel, const uint16_t input_width,
                         fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride) {

    uint16_t input_channel_idx;
    uint16_t output_channel_idx;
    uint16_t output_channel_width;
    uint16_t kernel_idx;

    output_channel_idx = 0;
    output_channel_width = (input_width-kernel_size)/stride+1;

    for (input_channel_idx = 0;
         input_channel_idx < input_width && output_channel_idx < output_channel_width; input_channel_idx += stride) {
        fp_t pixel = input_channel[input_channel_idx];

        for (kernel_idx = input_channel_idx;
             kernel_idx < input_channel_idx + kernel_size && kernel_idx < input_width; kernel_idx++) {
            if (input_channel[kernel_idx] > pixel) {
                pixel = input_channel[kernel_idx];
            }
        }
        output_channel[output_channel_idx] = pixel;
        output_channel_idx++;
    }
}

void max_pooling1d_naive_padded(const fp_t* input_channel, const uint16_t input_width,
                                fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                                const uint16_t* padding) {
    fp_t* new_input_channel;

    extend_1d_input_with_padding(input_channel, input_width, &new_input_channel, padding, FLOAT_MIN);

    max_pooling1d_naive(new_input_channel, input_width+padding[0]+padding[1],
                        output_channel, kernel_size, stride);

    free(new_input_channel);
}

void max_pooling2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                         fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride) {

    uint16_t channel_row, channel_column;
    uint16_t output_channel_row, output_channel_column;
    uint16_t output_channel_height, output_channel_width;

    uint16_t kernel_row, kernel_column;

    output_channel_row = 0;
    output_channel_column = 0;

    output_channel_height = (height-kernel_size)/stride+1;
    output_channel_width = (width-kernel_size)/stride+1;

    for (channel_row = 0; channel_row < height && output_channel_row < output_channel_height; channel_row += stride) {
        for (channel_column = 0;
             channel_column < width && output_channel_column < output_channel_width; channel_column += stride) {
            fp_t pixel = input_channel[channel_row * width + channel_column];

            for (kernel_row = channel_row;
                 kernel_row < channel_row + kernel_size && kernel_row < height; kernel_row++) {
                for (kernel_column = channel_column;
                     kernel_column < channel_column + kernel_size && kernel_column < width; kernel_column++) {
                    if (input_channel[kernel_row * width + kernel_column] > pixel) {
                        pixel = input_channel[kernel_row * width + kernel_column];
                    }
                }
            }

            output_channel[output_channel_row * output_channel_width + output_channel_column] = pixel;
            output_channel_column++;
        }
        output_channel_row++;
        output_channel_column = 0;
    }
}

void max_pooling2d_naive_padded(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                                fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                                const uint16_t* padding) {

    fp_t* new_input_channel;
    extend_2d_input_with_padding(input_channel, height, width, &new_input_channel, padding, FLOAT_MIN);

    max_pooling2d_naive(new_input_channel, height+padding[0]+padding[2], width+padding[1]+padding[3],
                        output_channel, kernel_size, stride);

    free(new_input_channel);
}

void average_pooling2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                             fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                             fp_t bias, const uint16_t count_include_pad) {

    uint16_t channel_row, channel_column;
    uint16_t output_channel_row, output_channel_column;
    uint16_t output_channel_height, output_channel_width;

    uint16_t kernel_row, kernel_column;

    output_channel_row = 0;
    output_channel_column = 0;

    output_channel_height = (height-kernel_size)/stride+1;
    output_channel_width = (width-kernel_size)/stride+1;

    if(count_include_pad == 1) {

        for (channel_row = 0;
             channel_row < height && output_channel_row < output_channel_height; channel_row += stride) {
            for (channel_column = 0;
                 channel_column < width && output_channel_column < output_channel_width; channel_column += stride) {
                fp_t pixel = 0.0;

                for (kernel_row = channel_row;
                     kernel_row < channel_row + kernel_size && kernel_row < height; kernel_row++) {
                    for (kernel_column = channel_column;
                         kernel_column < channel_column + kernel_size && kernel_column < width; kernel_column++) {
                        pixel += input_channel[kernel_row * width + kernel_column];
                    }
                }

                output_channel[output_channel_row * output_channel_width + output_channel_column] =
                        pixel / ((fp_t)(kernel_size * kernel_size)) + bias;
                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }

    } else if(count_include_pad == 0) {

        uint16_t crop = kernel_size/2;

        output_channel_row = 0;
        output_channel_column = 0;


        for (channel_row = 0;
             channel_row < height && output_channel_row < output_channel_height; channel_row += stride) {
            for (channel_column = 0;
                 channel_column < width && output_channel_column < output_channel_width; channel_column += stride) {
                fp_t pixel = 0.0;

                for (kernel_row = channel_row;
                     kernel_row < channel_row + kernel_size && kernel_row < height; kernel_row++) {
                    for (kernel_column = channel_column;
                         kernel_column < channel_column + kernel_size && kernel_column < width; kernel_column++) {
                        pixel += input_channel[kernel_row * width + kernel_column];
                    }
                }

                // center case
                if(output_channel_row >= crop && output_channel_row < output_channel_height-crop &&
                   output_channel_column >= crop && output_channel_column < output_channel_width-crop){

                    output_channel[output_channel_row * output_channel_width + output_channel_column] =
                            pixel / ((fp_t)(kernel_size * kernel_size)) + bias;

                // edge case
                } else {

                    uint16_t up_row, down_row, left_col, right_col;
                    int32_t divisor, div_row, div_col;

                    up_row = MAX(channel_row, crop);
                    down_row = MIN(channel_row+kernel_size-1, height-crop-1);
                    div_row = down_row-up_row+1;

                    left_col = MAX(channel_column, crop);
                    right_col = MIN(channel_column+kernel_size-1, width-crop-1);
                    div_col = right_col-left_col+1;

                    divisor = div_row * div_col;

                    if(divisor == 0) {
                        ERROR_MSG("ERROR: Division by zero! Aborting execution.\n");
                        exit(1);
                    }

                    output_channel[output_channel_row * output_channel_width + output_channel_column] =
                            pixel / ((fp_t) (divisor)) + bias;
                }


                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    } else {
        ERROR_MSG("ERROR: Unsupported values for 'count_include_pad'.\n");
    }
}

void average_pooling2d_naive_padded(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                                    fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                                    fp_t bias, const uint16_t* padding, const uint16_t count_include_pad) {

    fp_t* new_input_channel;
    extend_2d_input_with_padding(input_channel, height, width, &new_input_channel, padding, 0.0);

    average_pooling2d_naive(new_input_channel, height+padding[0]+padding[2], width+padding[1]+padding[3],
                            output_channel, kernel_size, stride, bias, count_include_pad);

    free(new_input_channel);
}

void average_pooling1d_naive(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel,
                             const uint16_t kernel_size, const uint16_t stride, fp_t bias,
                             const uint16_t count_include_pad) {

    uint16_t input_channel_idx;
    uint16_t output_channel_idx;
    uint16_t output_channel_width;
    uint16_t kernel_idx;

    output_channel_idx = 0;
    output_channel_width = (input_width-kernel_size)/stride+1;

    if(count_include_pad == 1) {

        for (input_channel_idx = 0;
             input_channel_idx < input_width && output_channel_idx < output_channel_width; input_channel_idx += stride) {

                fp_t pixel = 0.0;

                for (kernel_idx = input_channel_idx;
                     kernel_idx < input_channel_idx + kernel_size && kernel_idx < input_width; kernel_idx++) {
                        pixel += input_channel[kernel_idx];
                }

                output_channel[output_channel_idx] = pixel / ((fp_t) kernel_size) + bias;
            output_channel_idx++;
        }

    } else if(count_include_pad == 0) {

        uint16_t crop = kernel_size/2;

        for (input_channel_idx = 0;
             input_channel_idx < input_width && output_channel_idx < output_channel_width; input_channel_idx += stride) {

            fp_t pixel = 0.0;

            for (kernel_idx = input_channel_idx;
                 kernel_idx < input_channel_idx + kernel_size && kernel_idx < input_width; kernel_idx++) {
                pixel += input_channel[kernel_idx];
            }

            // center case
            if(output_channel_idx >= crop && output_channel_idx < output_channel_width-crop){

             output_channel[output_channel_idx] = pixel / ((fp_t) kernel_size) + bias;

             // edge case
            } else {

                uint16_t left_col, right_col;
                int32_t divisor, div_col;

                left_col = MAX(input_channel_idx, crop);
                right_col = MIN(input_channel_idx+kernel_size-1, input_width-crop-1);
                div_col = right_col-left_col+1;

                divisor = div_col;

                output_channel[output_channel_idx] = pixel / ((fp_t) (divisor)) + bias;
            }

            output_channel_idx++;
        }
    } else {
        ERROR_MSG("ERROR: Unsupported values for 'count_include_pad'.\n");
    }
}

void average_pooling1d_naive_padded(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel,
                                    const uint16_t kernel_size, const uint16_t stride, fp_t bias,
                                    const uint16_t* padding, const uint16_t count_include_pad) {
    fp_t* new_input_channel;

    extend_1d_input_with_padding(input_channel, input_width, &new_input_channel, padding, 0.0);

    average_pooling1d_naive(new_input_channel, input_width+padding[0]+padding[1],
                            output_channel, kernel_size, stride, bias, count_include_pad);

    free(new_input_channel);
}

void global_average_pooling2d_naive(const fp_t* input_channel, const uint16_t input_height,
                                    const uint16_t input_width, fp_t* output_channel) {
    uint16_t pixel;
    fp_t global_sum = 0.0;

    for(pixel = 0; pixel < input_height * input_width; pixel++){
        global_sum += input_channel[pixel];
    }

  output_channel[0] = global_sum / (fp_t)(input_height*input_width);

 }


void global_max_pooling2d_naive(const fp_t* input_channel, const uint16_t input_height,
                                const uint16_t input_width, fp_t* output_channel) {
    uint16_t pixel;
    // assumes that input_channel holds at least 1 value
    fp_t global_maximum = input_channel[0];

    for(pixel = 1; pixel < input_height * input_width; pixel++){
        if(global_maximum < input_channel[pixel]){
            global_maximum = input_channel[pixel];
        }
    }

    output_channel[0] = global_maximum;
}

void pad_2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                  fp_t* output_channel, const uint16_t* padding, fp_t initializer) {

    uint16_t height_padded = height + padding[0] + padding[2];
    uint16_t width_padded = width + padding[1] + padding[3];


    for(uint16_t r = 0; r < height_padded; r++){
        for(uint16_t c = 0; c < width_padded; c++){
            output_channel[r*width_padded+c] = initializer;
        }
    }

    for (int16_t channel_row = 0; channel_row < height; channel_row++) {
        memcpy(output_channel + (channel_row + padding[0]) * width_padded + padding[1],
               input_channel + channel_row * width, width * sizeof(fp_t));
    }
}

void pad_1d_naive(const fp_t* input_channel, const uint16_t width,
                  fp_t* output_channel, const uint16_t* padding, fp_t initializer) {
    uint16_t width_padded = width + padding[0] + padding[1];

    for(uint16_t i = 0; i < width_padded; i++){
        output_channel[i] = initializer;
    }

    memcpy(output_channel+padding[0], input_channel, width*sizeof(fp_t));
}
