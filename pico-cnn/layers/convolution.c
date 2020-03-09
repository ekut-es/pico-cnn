#include "convolution.h"

void convolution1d_naive(const fp_t* input_channel, const uint16_t input_size, fp_t* output_channel, const fp_t* kernel,
                         const uint16_t kernel_size, const uint16_t stride, const uint16_t padding, const fp_t bias) {

    int32_t input_channel_idx;
    int32_t kernel_idx;
    int32_t crop = kernel_size/2;
    int32_t output_channel_idx;

    fp_t pixel;

    output_channel_idx = 0;

    // padding valid
    if(padding == 0) {

        for(input_channel_idx = crop; input_channel_idx < input_size-crop; input_channel_idx+=stride) {
            pixel = 0.0;

            for(kernel_idx = 0; kernel_idx < kernel_size; kernel_idx++) {
                pixel += kernel[kernel_idx] * input_channel[input_channel_idx-crop+kernel_idx];
            }

            pixel += bias;

            output_channel[output_channel_idx] = pixel;
            output_channel_idx++;
        }
    }

    // padding same
    else if(padding == kernel_size/2) {

        for(input_channel_idx = 0; input_channel_idx < input_size; input_channel_idx+=stride) {
            pixel = 0.0;
            for(kernel_idx = -padding; kernel_idx <= padding; kernel_idx++) {
                if((input_channel_idx+kernel_idx) < 0 || (input_channel_idx+kernel_idx) > input_size-1) {
                    pixel += 0.0;
                } else {
                    pixel += kernel[kernel_idx+padding] * input_channel[input_channel_idx+kernel_idx];
                }
            }
            pixel += bias;

            output_channel[output_channel_idx] = pixel;
            output_channel_idx++;
        }
    }
}

void convolution2d_padding_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel,
                                 const fp_t* kernel, const uint16_t kernel_height, const uint16_t kernel_width,
                                 const uint16_t stride_height, const uint16_t stride_width,
                                 const uint16_t* padding, const fp_t bias) {

   fp_t* new_input_channel;

   extend_2d_input_with_padding(input_channel, height, width, &new_input_channel, padding, 0);

   convolution2d_naive(new_input_channel, height+padding[0]+padding[2], width+padding[1]+padding[3],
                                   output_channel, kernel, kernel_height, kernel_width,
                                   stride_height, stride_width, bias);

   free(new_input_channel);
}

void convolution2d_naive(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel,
                         const fp_t* kernel, const uint16_t kernel_height, const uint16_t kernel_width,
                         const uint16_t stride_height, const uint16_t stride_width, const fp_t bias) {


   int32_t channel_row, channel_column;
   int32_t kernel_row, kernel_column;
   int32_t height_crop = kernel_height/2;
   int32_t width_crop = kernel_width/2;

   fp_t pixel;

   int32_t output_channel_row = 0;
   int32_t output_channel_column = 0;

   uint16_t output_channel_width = (width - kernel_width)/stride_width + 1;


   for(channel_row = height_crop; channel_row < height-height_crop; channel_row+=stride_height) {
       for(channel_column = width_crop; channel_column < width - width_crop; channel_column += stride_width) {
           pixel = 0.0;

           for(kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
               for(kernel_column = 0; kernel_column < kernel_width; kernel_column++) {
                   pixel += kernel[kernel_row*kernel_width + kernel_column] *
                            input_channel[width*(channel_row-height_crop+kernel_row) + channel_column-width_crop+kernel_column];
               }
           }

           pixel += bias;
           output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
           output_channel_column++;

       }
       output_channel_row++;
       output_channel_column = 0;

   }

}

void convolution2d_naive_legacy(const fp_t* input_channel, const uint16_t height, const uint16_t width, fp_t* output_channel,
                                const fp_t* kernel, const uint16_t kernel_size, const uint16_t stride,
                                const uint16_t padding, const fp_t bias) {

    int32_t channel_row, channel_column;
    int32_t kernel_row, kernel_column;
    int32_t crop = kernel_size/2;

    int32_t output_channel_row, output_channel_column, output_channel_width;

    fp_t pixel;

    output_channel_row = 0;
    output_channel_column = 0;

    // padding valid
    if(padding == 0) {
        output_channel_width = (width-kernel_size)/stride+1;

        for(channel_row = crop; channel_row < height-crop; channel_row+=stride) {
            for(channel_column = crop; channel_column < width-crop; channel_column+=stride) {
                pixel = 0.0;

                for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                        pixel += kernel[kernel_row*kernel_size+kernel_column] * input_channel[(channel_row-crop+kernel_row)*width+(channel_column-crop+kernel_column)];
                    }
                }

                pixel += bias;

                output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    }

    // padding same
    else if(padding == kernel_size/2) {
        output_channel_width = (width+2*(kernel_size/2)-kernel_size)/stride+1;

        for(channel_row = 0; channel_row < height; channel_row+=stride) {
            for(channel_column = 0; channel_column < width; channel_column+=stride) {
                pixel = 0.0;

                for(kernel_row = -padding; kernel_row <= padding; kernel_row++) {
                    for(kernel_column = -padding; kernel_column <= padding; kernel_column++) {
                        if((channel_row+kernel_row) < 0 || (channel_row+kernel_row) > height-1 || (channel_column+kernel_column) < 0 || (channel_column+kernel_column) > width-1) {
                            pixel += 0.0;
                        } else {
                            pixel += kernel[(kernel_row+padding)*kernel_size+(kernel_column+padding)] * input_channel[(channel_row+kernel_row)*width+(channel_column+kernel_column)];
                        }
                    }
                }

                pixel += bias;

                output_channel[output_channel_row*output_channel_width+output_channel_column] = pixel;
                output_channel_column++;
            }
            output_channel_row++;
            output_channel_column = 0;
        }
    }
}

void add_channel2d_naive(fp_t* channel_a, const fp_t* channel_b, const uint16_t height, const uint16_t width) {
    uint32_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            channel_a[row*width+column] = (channel_a[row*width+column] + channel_b[row*width+column]);
        }
    }
}
