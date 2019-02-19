/** 
 * @brief contains all convolutions
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif 


/**
 * @brief performs a 2D convolution on original_image with kernel and stores the
 * result to new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_size x kernel_size)
 * @param kernel_size
 * @param stride
 * @param padding (0 means valid, > 0 zeros will be added to the edge)
 * @param bias
 */
void convolution2d_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const uint16_t kernel_size, const uint16_t stride, const uint16_t padding, const fp_t bias) {
    int32_t image_row, image_column;
    int32_t kernel_row, kernel_column;
    int32_t crop = kernel_size/2;

    int32_t new_image_row, new_image_column, new_image_width;

    fp_t pixel;

    new_image_row = 0;
    new_image_column = 0;

    // padding valid
    if(padding == 0) {
        if(stride == 1) {
            new_image_width = ((width-2*crop)/stride);
        } else {
            new_image_width = ((width-2*crop)/stride)+1;
        }

        for(image_row = crop; image_row < height-crop; image_row+=stride) {
            for(image_column = crop; image_column < width-crop; image_column+=stride) {
                pixel = 0.0;

                for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                        pixel += kernel[kernel_row*kernel_size+kernel_column] * original_image[(image_row-crop+kernel_row)*width+(image_column-crop+kernel_column)];
                    }
                }

                pixel += bias;

                new_image[new_image_row*new_image_width+new_image_column] = pixel;
                new_image_column++;
            }
            new_image_row++;
            new_image_column = 0;
        }
    }

    // padding same
    else if(padding == kernel_size/2) {
        new_image_width = width;

        for(image_row = 0; image_row < height; image_row+=stride) {
            for(image_column = 0; image_column < width; image_column+=stride) {
                pixel = 0.0;

                for(kernel_row = -padding; kernel_row <= padding; kernel_row++) {
                    for(kernel_column = -padding; kernel_column <= padding; kernel_column++) {
                        if((image_row+kernel_row) < 0 || (image_row+kernel_row) > height-1 || (image_column+kernel_column) < 0 || (image_column+kernel_column) > width-1) {
                            pixel += 0.0;
                        } else {
                            pixel += kernel[(kernel_row+padding)*kernel_size+(kernel_column+padding)] * original_image[(image_row+kernel_row)*width+(image_column+kernel_column)];
                        }
                    }
                } 

                pixel += bias;

                new_image[new_image_row*new_image_width+new_image_column] = pixel;
                new_image_column++;
            }
            new_image_row++;
            new_image_column = 0;
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
    uint32_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            image_a[row*width+column] = (image_a[row*width+column] + image_b[row*width+column]);
        }
    }
}

#ifdef FIXED16
/**
 * @brief performs a 2D convolution on original_image with kernel and stores the
 * result to new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height-kernel_size/2 x width-kernel_size/2)
 * @param kernel (kernel_size x kernel_size)
 * @param kernel_size
 * @param stride
 * @param padding (0 means valid, > 0 zeros will be added to the edge)
 * @param bias
 */
void convolution2d_naive_fixed16(const fixed16_t* original_image, const uint16_t height, const uint16_t width, fixed16_t* new_image, const fixed16_t* kernel, const uint16_t kernel_size, const uint16_t stride, const uint16_t padding, const fixed16_t bias) {
    int32_t image_row, image_column;
    int32_t kernel_row, kernel_column;
    int32_t crop = kernel_size/2;

    int32_t new_image_row, new_image_column, new_image_width;

    fixed16_t pixel;

    new_image_row = 0;
    new_image_column = 0;

    // padding valid
    if(padding == 0) {
        if(stride == 1) {
            new_image_width = ((width-2*crop)/stride);
        } else {
            new_image_width = ((width-2*crop)/stride)+1;
        }

        for(image_row = crop; image_row < height-crop; image_row+=stride) {
            for(image_column = crop; image_column < width-crop; image_column+=stride) {
                pixel = 0;

                for(kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    for(kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                        pixel = add_fixed16(pixel, mul_fixed16(kernel[kernel_row*kernel_size+kernel_column], original_image[(image_row-crop+kernel_row)*width+(image_column-crop+kernel_column)]));
                    }
                }

                pixel = add_fixed16(pixel, bias);

                new_image[new_image_row*new_image_width+new_image_column] = pixel;
                new_image_column++;
            }
            new_image_row++;
            new_image_column = 0;
        }
    }

    // padding same
    else if(padding == kernel_size/2) {
        new_image_width = width;

        for(image_row = 0; image_row < height; image_row+=stride) {
            for(image_column = 0; image_column < width; image_column+=stride) {
                pixel = 0;

                for(kernel_row = -padding; kernel_row <= padding; kernel_row++) {
                    for(kernel_column = -padding; kernel_column <= padding; kernel_column++) {
                        if((image_row+kernel_row) < 0 || (image_row+kernel_row) > height-1 || (image_column+kernel_column) < 0 || (image_column+kernel_column) > width-1) {
                            pixel = add_fixed16(0, pixel);
                        } else {
                            pixel = add_fixed16(pixel, mul_fixed16(kernel[(kernel_row+padding)*kernel_size+(kernel_column+padding)], original_image[(image_row+kernel_row)*width+(image_column+kernel_column)]));
                        }
                    }
                } 

                pixel = add_fixed16(pixel, bias);

                new_image[new_image_row*new_image_width+new_image_column] = pixel;
                new_image_column++;
            }
            new_image_row++;
            new_image_column = 0;
        }
    }
}

void convolution2d_cpu_5x5_s1_valid_fixed16(const fixed16_t* original_image, const uint16_t height, const uint16_t width, fixed16_t* new_image, const fixed16_t* kernel, const fixed16_t bias) {
	convolution2d_naive_fixed16(original_image, height, width, new_image, kernel, 5, 1, 0, bias);
}

/**
 * @brief adds image_a and image_b pixel by pixel and stores result in image_a
 * 
 * @param image_a (height x width)
 * @param image_b (height x width)
 * @param height
 * @param width
 */
void add_image2d_naive_fixed16(fixed16_t* image_a, const fixed16_t* image_b, const uint16_t height, const uint16_t width) {
    uint32_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            image_a[row*width+column] = add_fixed16(image_a[row*width+column], image_b[row*width+column]);
        }
    }
}

void add_image2d_cpu_fixed16(fixed16_t* image_a, const fixed16_t* image_b, const uint16_t height, const uint16_t width) {
	add_image2d_naive_fixed16(image_a, image_b, height, width);
}
#endif

#endif // CONVOLUTION_H
