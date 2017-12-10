/** 
 * @brief contains all poolings
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef POOLING_H
#define POOLING_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>

/**
 * @brief applies max pooling of kernel_size x kernel_size to original_image 
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height/kernel_size x width/kernel_size)
 * @param kernel_size
 * @param stride
 */
void max_pooling2d_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const uint16_t kernel_size, const uint16_t stride) {

    uint16_t image_row, image_column;
    uint16_t new_image_row, new_image_column;
    uint16_t new_image_height, new_image_width;

    uint16_t kernel_row, kernel_column;

    new_image_row = 0;
    new_image_column = 0;
    
    new_image_height = height/stride;
    new_image_width = width/stride;

    for(image_row = 0; image_row < height && new_image_row < new_image_height; image_row += stride) {
        for(image_column = 0; image_column < width && new_image_column < new_image_width; image_column += stride) {
            fp_t pixel = original_image[image_row*width+image_column];
    
            for(kernel_row = image_row; kernel_row < image_row+kernel_size && kernel_row < height; kernel_row++) {
                for(kernel_column = image_column; kernel_column < image_column+kernel_size && kernel_column < width; kernel_column++) {
                    if(original_image[kernel_row*width+kernel_column] > pixel) {
                        pixel = original_image[kernel_row*width+kernel_column];
                    }
                }
            }
            
            new_image[new_image_row*new_image_width+new_image_column] = pixel;
            new_image_column++;
        }
        new_image_row++;
        new_image_column = 0;
    }
}

/**
 * @brief applies average pooling of kernel_size x kernel_size to original_image 
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void average_pooling2d_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const uint16_t kernel_size, fp_t bias) {

    uint16_t row, column;

    for(row = 0; row < height; row += kernel_size) {
        for(column = 0; column < width; column += kernel_size) {
            fp_t pixel = original_image[row*width+column];
    
            uint16_t sub_row, sub_column;
            
            for(sub_row = row; sub_row < row+kernel_size; sub_row++) {
                for(sub_column = column; sub_column < column+kernel_size; sub_column++) {
                    pixel += original_image[sub_row*width+sub_column];
                }
            }
            
            new_image[(row/kernel_size)*(height/kernel_size)+(column/kernel_size)] = pixel/((fp_t) kernel_size*kernel_size) + bias;
        }
    }
}

#ifdef __aarch64__
/**
 * @brief applies max pooling of kernel_size x kernel_size to original_image 
 *
 * kernel_size = 2
 * stride = 2
 *
 * @param original_image (height x width)
 * @param new_image (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void max_pooling2d_cpu_2x2_s2(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t kernel_size = 2;
    uint16_t stride = 2;

    uint16_t image_row, image_column;
    uint16_t new_image_row, new_image_column;
    uint16_t new_image_height, new_image_width;

    uint16_t kernel_row, kernel_column;

    new_image_row = 0;
    new_image_column = 0;
    
    new_image_height = height/stride;
    new_image_width = width/stride;

    for(image_row = 0; image_row < height && new_image_row < new_image_height; image_row += stride) {
        for(image_column = 0; image_column < width && new_image_column < new_image_width; image_column += stride) {
            fp_t pixel = original_image[image_row*width+image_column];
    
            for(kernel_row = image_row; kernel_row < image_row+kernel_size && kernel_row < height; kernel_row++) {
                for(kernel_column = image_column; kernel_column < image_column+kernel_size && kernel_column < width; kernel_column++) {
                    if(original_image[kernel_row*width+kernel_column] > pixel) {
                        pixel = original_image[kernel_row*width+kernel_column];
                    }
                }
            }
            
            new_image[new_image_row*new_image_width+new_image_column] = pixel;
            new_image_column++;
        }
        new_image_row++;
        new_image_column = 0;
    }
}


/**
 * @brief applies max pooling of kernel_size x kernel_size to original_image 
 *
 * kernel_size = 3
 * stride = 2
 *
 * @param original_image (height x width)
 * @param new_image (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void max_pooling2d_cpu_3x3_s2(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t kernel_size = 3;
    uint16_t stride = 2;

    uint16_t image_row, image_column;
    uint16_t new_image_row, new_image_column;
    uint16_t new_image_height, new_image_width;

    uint16_t kernel_row, kernel_column;

    new_image_row = 0;
    new_image_column = 0;
    
    new_image_height = height/stride;
    new_image_width = width/stride;

    for(image_row = 0; image_row < height && new_image_row < new_image_height; image_row += stride) {
        for(image_column = 0; image_column < width && new_image_column < new_image_width; image_column += stride) {
            fp_t pixel = original_image[image_row*width+image_column];
    
            for(kernel_row = image_row; kernel_row < image_row+kernel_size && kernel_row < height; kernel_row++) {
                for(kernel_column = image_column; kernel_column < image_column+kernel_size && kernel_column < width; kernel_column++) {
                    if(original_image[kernel_row*width+kernel_column] > pixel) {
                        pixel = original_image[kernel_row*width+kernel_column];
                    }
                }
            }
            
            new_image[new_image_row*new_image_width+new_image_column] = pixel;
            new_image_column++;
        }
        new_image_row++;
        new_image_column = 0;
    }
}
#endif

#endif // POOLING_H
