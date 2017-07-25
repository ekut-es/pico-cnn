#include "parameters.h"
#include <stdint.h>
#include <stdio.h>

// shrinks image!
void convolution2d(const float_t* original_image, const uint16_t height, const uint16_t width, float_t* new_image, const float_t* kernel, const uint16_t kernel_size, const uint16_t stride) {
    uint16_t image_row, image_column;
    uint16_t kernel_row, kernel_column;
    uint8_t padding = kernel_size/2;

    float_t pixel;
    
    for(image_row = padding; image_row < height-padding; image_row++) {
        for(image_column = padding; image_column < width-padding; image_column++) {
            pixel = 0.0;

            for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                    pixel += kernel[kernel_row*kernel_size+kernel_column] * original_image[(image_row-padding+kernel_row)*width+(image_column-padding+kernel_column)];
                }
            }

            new_image[(image_row-padding)*(width-2*padding)+(image_column-padding)] = pixel;
        }
    }
}
