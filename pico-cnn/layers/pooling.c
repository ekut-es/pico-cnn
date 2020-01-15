#include "pooling.h"

void extend_2d_input_with_padding(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                               fp_t** extended_input, const int* padding, fp_t initializer) {

    uint16_t height_padded = height + padding[0] + padding[2];
    uint16_t width_padded = width + padding[1] + padding[3];

    *extended_input = (fp_t*) calloc(height_padded * width_padded, sizeof(fp_t));

    if(initializer != 0.0) {
        for(int r = 0; r < height_padded; r++){
            for(int c = 0; c < width_padded; c++){
                (*extended_input)[r*width_padded+c] = initializer;
            }
        }
    }

    for (int16_t channel_row = 0; channel_row < height; channel_row++) {
        memcpy((*extended_input) + (channel_row + padding[0]) * width_padded + padding[1],
               input_channel + channel_row * width, width * sizeof(fp_t));
    }
}

void extend_1d_input_with_padding(const fp_t* input_channel, const uint16_t width,
                                  fp_t** extended_input, const int* padding, fp_t initializer) {

    uint16_t width_padded = width + padding[0] + padding[1];

    *extended_input = (fp_t*) calloc(width_padded, sizeof(fp_t));

    if(initializer != 0.0) {
        for(int i = 0; i < width_padded; i++){
            (*extended_input)[i] = initializer;
        }
    }

    memcpy((*extended_input)+padding[0], input_channel, width*sizeof(fp_t));
}

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
                                const int* padding) {
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
                                const int* padding) {

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
                        pixel / ((fp_t) kernel_size * kernel_size) + bias;
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

//                    printf("Center case: output_row: %d, output_column: %d\n", output_channel_row, output_channel_column);

                    output_channel[output_channel_row * output_channel_width + output_channel_column] =
                            pixel / ((fp_t)(kernel_size * kernel_size)) + bias;

                // edge case
                } else {

                    int orig_height = height-2*crop;
                    int orig_width = width-2*crop;

                    int up_row, down_row, left_col, right_col;
                    int divisor, div_row, div_col;

                    up_row = MAX(channel_row, crop);
                    down_row = MIN(channel_row+kernel_size-1, height-crop-1);
                    div_row = down_row-up_row+1;

                    left_col = MAX(channel_column, crop);
                    right_col = MIN(channel_column+kernel_size-1, width-crop-1);
                    div_col = right_col-left_col+1;

                    divisor = div_row * div_col;

//                    printf("Edge case: output_row: %d, output_column: %d\n", output_channel_row, output_channel_column);
//                    printf("div_row: %d, div_col: %d\n", div_row, div_col);
//                    printf("Divisor: %d\n", divisor);

                    output_channel[output_channel_row * output_channel_width + output_channel_column] =
                            pixel / ((fp_t) (divisor)) + bias;
                }


                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    } else {
        printf( "ERROR: Unsupported values for 'count_include_pad'.\n");
    }
}

void average_pooling2d_naive_padded(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                                    fp_t* output_channel, const uint16_t kernel_size, const uint16_t stride,
                                    fp_t bias, const int* padding, const uint16_t count_include_pad) {

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

                int orig_width = input_width-2*crop;

                int left_col, right_col;
                int divisor, div_col;

                left_col = MAX(input_channel_idx, crop);
                right_col = MIN(input_channel_idx+kernel_size-1, input_width-crop-1);
                div_col = right_col-left_col+1;

                divisor = div_col;

                output_channel[output_channel_idx] = pixel / ((fp_t) (divisor)) + bias;
            }

            output_channel_idx++;
        }
    } else {
        printf( "ERROR: Unsupported values for 'count_include_pad'.\n");
    }
}

void average_pooling1d_naive_padded(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel,
                                    const uint16_t kernel_size, const uint16_t stride, fp_t bias, const int* padding,
                                    const uint16_t count_include_pad) {
    fp_t* new_input_channel;

    extend_1d_input_with_padding(input_channel, input_width, &new_input_channel, padding, 0.0);

    average_pooling1d_naive(new_input_channel, input_width+padding[0]+padding[1],
                            output_channel, kernel_size, stride, bias, count_include_pad);

    free(new_input_channel);
}

void global_average_pooling2d_naive(const fp_t* input_channel, const uint16_t input_width,
                                    const uint16_t input_height, fp_t* output_channel) {
    uint16_t pixel;
    fp_t global_sum = 0;

    for(pixel = 0; pixel < input_height * input_width; pixel++){
        global_sum += input_channel[pixel];
    }

  output_channel[0] = global_sum / (input_height*input_width);

 }


 void global_max_pooling2d_naive(const fp_t* input_channel, const uint16_t input_width,
                                 const uint16_t input_height, fp_t* output_channel) {
     output_channel[0] = 0;

 }


#ifdef FIXED16

void max_pooling2d_naive_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width,
                                 fixed16_t* output_channel, const uint16_t kernel_size, const uint16_t stride) {

    uint16_t channel_row, channel_column;
    uint16_t output_channel_row, output_channel_column;
    uint16_t output_channel_height, output_channel_width;

    uint16_t kernel_row, kernel_column;

    output_channel_row = 0;
    output_channel_column = 0;

    output_channel_height = height/stride;
    output_channel_width = width/stride;

    for(channel_row = 0; channel_row < height && output_channel_row < output_channel_height; channel_row += stride) {
        for(channel_column = 0; channel_column < width && output_channel_column < output_channel_width; channel_column += stride) {
            fp_t pixel = input_channel[channel_row*width+channel_column];

            for(kernel_row = channel_row; kernel_row < channel_row+kernel_size && kernel_row < height; kernel_row++) {
                for(kernel_column = channel_column; kernel_column < channel_column+kernel_size && kernel_column < width; kernel_column++) {
                    if(input_channel[kernel_row*width+kernel_column] > pixel) {
                        pixel = input_channel[kernel_row*width+kernel_column];
                    }
                }
            }

            output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
            output_channel_column++;
        }
        output_channel_row++;
        output_channel_column = 0;
    }
}

/**
 * @brief TODO
 */
void max_pooling2d_cpu_2x2_s2_fixed16(const fixed16_t* input_channel, const uint16_t height, const uint16_t width,
                                      fixed16_t* output_channel) {
    max_pooling2d_naive_fixed16(input_channel, height, width, output_channel, 2, 2);
}

#endif // FIXED16

#ifdef ARM_NEON

void max_pooling2d_cpu_2x2_s2(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                              fp_t* output_channel) {

    uint16_t stride = 2;

    uint16_t channel_row, channel_column;
    uint16_t output_channel_row, output_channel_column;
    uint16_t output_channel_width;

    output_channel_row = 0;
    output_channel_column = 0;

    output_channel_width = width/stride;


    float32x4_t input_channel_0;
    float32x4_t input_channel_1;
    float32x4_t input_channel_2;
    float32x4_t input_channel_3;
    float32x4_t input_channel_4;
    float32x4_t input_channel_5;
    float32x4_t input_channel_6;
    float32x4_t input_channel_7;

    float32x2_t temp_max_0;
    float32x2_t temp_max_1;
    float32x2_t temp_max_2;
    float32x2_t temp_max_3;
    float32x2_t temp_max_4;
    float32x2_t temp_max_5;
    float32x2_t temp_max_6;
    float32x2_t temp_max_7;

    fp_t pixel_0;
    fp_t pixel_1;
    fp_t pixel_2;
    fp_t pixel_3;
    fp_t pixel_4;
    fp_t pixel_5;
    fp_t pixel_6;
    fp_t pixel_7;


    for(channel_row = 0; channel_row < height-4; channel_row += 4) {
        for(channel_column = 0; channel_column < width-8; channel_column += 8) {

            const uint32_t source_0 = (channel_row*width)+channel_column;
            const uint32_t source_1 = ((channel_row+1)*width)+channel_column;
            const uint32_t source_2 = ((channel_row+2)*width)+channel_column;
            const uint32_t source_3 = ((channel_row+3)*width)+channel_column;

            // load channel into vectors
            input_channel_0 = vld1q_f32(input_channel+source_0);
            input_channel_1 = vld1q_f32(input_channel+source_1);
            input_channel_2 = vld1q_f32(input_channel+source_2);
            input_channel_3 = vld1q_f32(input_channel+source_3);

            input_channel_4 = vld1q_f32(input_channel+source_0+4);
            input_channel_5 = vld1q_f32(input_channel+source_1+4);
            input_channel_6 = vld1q_f32(input_channel+source_2+4);
            input_channel_7 = vld1q_f32(input_channel+source_3+4);

            // determine max of halfs
            temp_max_0 = vpmax_f32(vget_low_f32(input_channel_0), vget_low_f32(input_channel_1));
            temp_max_1 = vpmax_f32(vget_high_f32(input_channel_0), vget_high_f32(input_channel_1));
            temp_max_2 = vpmax_f32(vget_low_f32(input_channel_2), vget_low_f32(input_channel_3));
            temp_max_3 = vpmax_f32(vget_high_f32(input_channel_2), vget_high_f32(input_channel_3));

            temp_max_4 = vpmax_f32(vget_low_f32(input_channel_4), vget_low_f32(input_channel_5));
            temp_max_5 = vpmax_f32(vget_high_f32(input_channel_4), vget_high_f32(input_channel_5));
            temp_max_6 = vpmax_f32(vget_low_f32(input_channel_6), vget_low_f32(input_channel_7));
            temp_max_7 = vpmax_f32(vget_high_f32(input_channel_6), vget_high_f32(input_channel_7));


            // determine max of temp_max_*
            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);
            temp_max_2 = vpmax_f32(temp_max_2, temp_max_2);
            temp_max_3 = vpmax_f32(temp_max_3, temp_max_3);

            temp_max_4 = vpmax_f32(temp_max_4, temp_max_4);
            temp_max_5 = vpmax_f32(temp_max_5, temp_max_5);
            temp_max_6 = vpmax_f32(temp_max_6, temp_max_6);
            temp_max_7 = vpmax_f32(temp_max_7, temp_max_7);

            pixel_0 = vget_lane_f32(temp_max_0, 0);
            pixel_1 = vget_lane_f32(temp_max_1, 0);
            pixel_2 = vget_lane_f32(temp_max_2, 0);
            pixel_3 = vget_lane_f32(temp_max_3, 0);

            pixel_4 = vget_lane_f32(temp_max_4, 0);
            pixel_5 = vget_lane_f32(temp_max_5, 0);
            pixel_6 = vget_lane_f32(temp_max_6, 0);
            pixel_7 = vget_lane_f32(temp_max_7, 0);

            const uint32_t target_0 = output_channel_row*output_channel_width+output_channel_column;
            const uint32_t target_1 = (output_channel_row+1)*output_channel_width+output_channel_column;

            output_channel[target_0] = pixel_0;
            output_channel[target_0+1] = pixel_1;
            output_channel[target_1] = pixel_2;
            output_channel[target_1+1] = pixel_3;

            output_channel[target_0+2] = pixel_4;
            output_channel[target_0+3] = pixel_5;
            output_channel[target_1+2] = pixel_6;
            output_channel[target_1+3] = pixel_7;

            output_channel_column+=4;
        }

        // residual columns
        for(channel_column = channel_column; channel_column < width; channel_column+=2) {

            temp_max_0 = vld1_f32(input_channel+(channel_row*width)+channel_column);
            temp_max_1 = vld1_f32(input_channel+((channel_row+1)*width)+channel_column);
            temp_max_2 = vld1_f32(input_channel+((channel_row+2)*width)+channel_column);
            temp_max_3 = vld1_f32(input_channel+((channel_row+3)*width)+channel_column);

            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);
            temp_max_2 = vpmax_f32(temp_max_2, temp_max_2);
            temp_max_3 = vpmax_f32(temp_max_3, temp_max_3);

            pixel_0 = MAX(vget_lane_f32(temp_max_0, 0), vget_lane_f32(temp_max_1, 0));
            pixel_1 = MAX(vget_lane_f32(temp_max_2, 0), vget_lane_f32(temp_max_3, 0));

            output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel_0;
            output_channel[(output_channel_row+1)*output_channel_width+output_channel_column] = pixel_1;

            output_channel_column++;
        }

        output_channel_row+=2;
        output_channel_column = 0;
    }

    // residual rows
    for(channel_row = channel_row; channel_row < height; channel_row+=2) {
        for(channel_column = 0; channel_column < width-4; channel_column += 4) {

            // load channel into vectors
            input_channel_0 = vld1q_f32(input_channel+(channel_row*width)+channel_column);
            input_channel_1 = vld1q_f32(input_channel+((channel_row+1)*width)+channel_column);

            // determine max of halfs
            temp_max_0 = vpmax_f32(vget_low_f32(input_channel_0), vget_low_f32(input_channel_1));
            temp_max_1 = vpmax_f32(vget_high_f32(input_channel_0), vget_high_f32(input_channel_1));

            // determine max of temp_max_*
            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);

            pixel_0 = vget_lane_f32(temp_max_0, 0);
            pixel_1 = vget_lane_f32(temp_max_1, 0);

            output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel_0;
            output_channel[output_channel_row*output_channel_width+output_channel_column+1] = pixel_1;

            output_channel_column+=2;
        }

        // residual columns
        for(channel_column = channel_column; channel_column < width; channel_column+=2) {

            temp_max_0 = vld1_f32(input_channel+(channel_row*width)+channel_column);
            temp_max_1 = vld1_f32(input_channel+((channel_row+1)*width)+channel_column);

            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);

            pixel_0 = MAX(vget_lane_f32(temp_max_0, 0), vget_lane_f32(temp_max_1, 0));

            output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel_0;
            output_channel_column++;
        }

        output_channel_row++;
        output_channel_column = 0;
    }
}

void max_pooling2d_cpu_3x3_s2(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel) {

    uint16_t stride = 2;

    uint16_t channel_row, channel_column;
    uint16_t output_channel_row, output_channel_column;
    uint16_t output_channel_height, output_channel_width;

    output_channel_row = 0;
    output_channel_column = 0;

    output_channel_height = height/stride;
    output_channel_width = width/stride;

    float32x4_t input_channel_0;
    float32x4_t input_channel_1;
    float32x4_t input_channel_2;

    for(channel_row = 0; channel_row < height && output_channel_row < output_channel_height; channel_row += 2) {
        for(channel_column = 0; channel_column < width && output_channel_column < output_channel_width; channel_column += 2) {

            const uint32_t source_0 = (channel_row*width)+channel_column;
            const uint32_t source_1 = ((channel_row+1)*width)+channel_column;
            const uint32_t source_2 = ((channel_row+2)*width)+channel_column;

            input_channel_0 = vld1q_f32(input_channel+source_0);
            input_channel_1 = vld1q_f32(input_channel+source_1);
            input_channel_2 = vld1q_f32(input_channel+source_2);

            input_channel_0 = vmaxq_f32(input_channel_0, input_channel_1);
            input_channel_0 = vmaxq_f32(input_channel_0, input_channel_2);

            fp_t max = -FLT_MAX;

            max = MAX(MAX(vgetq_lane_f32(input_channel_0, 0), vgetq_lane_f32(input_channel_0, 1)), vgetq_lane_f32(input_channel_0, 2));

            output_channel[output_channel_row*output_channel_width+output_channel_column] = max;
            output_channel_column++;
        }
        output_channel_row++;
        output_channel_column = 0;
    }
}
#endif // ARM_NEON
