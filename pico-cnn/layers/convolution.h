/** 
 * @brief contains all convolutions
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../parameters.h"
#include <stdint.h>

/**
 * @brief performs a 2D convolution on original_image with kernel and stores the
 * result to new_image
 *
 * stride = 1
 * padding = valid => imags shrinks by kernel_size/2
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_size x kernel_size)
 * @param kernel_size
 * @param bias
 */
void convolution2d_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const uint16_t kernel_size, const fp_t bias) {
    uint16_t image_row, image_column;
    uint16_t kernel_row, kernel_column;
    uint8_t padding = kernel_size/2;

    fp_t pixel;
    
    for(image_row = padding; image_row < height-padding; image_row++) {
        for(image_column = padding; image_column < width-padding; image_column++) {
            pixel = 0.0;

            for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                    pixel += kernel[kernel_row*kernel_size+kernel_column] * original_image[(image_row-padding+kernel_row)*width+(image_column-padding+kernel_column)];
                }
            }

            pixel += bias;

            new_image[(image_row-padding)*(width-2*padding)+(image_column-padding)] = pixel;
        }
    }
}

/**
 * @brief adds image_a and image_b pixel by pixel and stores result in image_a
 * 
 * @param image_a (height x width)
 * @param image_b (height x width)
 * @param height
 * @param width
 */
void add_image2d_naive(fp_t* image_a, const fp_t* image_b, const uint16_t height, const uint16_t width) {
    uint16_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            image_a[row*width+column] = (image_a[row*width+column] + image_b[row*width+column]);
        }
    }
}

/**
 * @brief performs an CPU optimized 2D convolution on original_image with kernel and stores the
 * result to new_image
 *
 * stride = 1
 * padding = valid => imags shrinks by kernel_size/2
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_size x kernel_size)
 * @param kernel_size
 * @param bias
 */
void convolution2d_cpu(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const uint16_t kernel_size, const fp_t bias) {
    uint16_t image_row, image_column;
    uint16_t kernel_row, kernel_column;
    uint8_t padding = kernel_size/2;

    fp_t pixel;
    
    for(image_row = padding; image_row < height-padding; image_row++) {
        for(image_column = padding; image_column < width-padding; image_column++) {
            pixel = 0.0;

            for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                    pixel += kernel[kernel_row*kernel_size+kernel_column] * original_image[(image_row-padding+kernel_row)*width+(image_column-padding+kernel_column)];
                }
            }

            pixel += bias;

            new_image[(image_row-padding)*(width-2*padding)+(image_column-padding)] = pixel;
        }
    }
}

/**
 * @brief adds image_a and image_b pixel by pixel and stores result in image_a optimized for CPU
 * 
 * @param image_a (height x width)
 * @param image_b (height x width)
 * @param height
 * @param width
 */
void add_image2d_cpu(fp_t* image_a, const fp_t* image_b, const uint16_t height, const uint16_t width) {
    uint16_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            image_a[row*width+column] = (image_a[row*width+column] + image_b[row*width+column]);
        }
    }
}
#endif // CONVOLUTION_H
