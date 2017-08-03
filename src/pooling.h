#include "parameters.h"
#include <stdint.h>

/**
 * @brief applies max pooling of kernel_size x kernel_size to original_image 
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height/kernel_size x width/kernel_size)
 * @param kernel_size
 */
void max_pooling2d_naive(const float_t* original_image, const uint16_t height, const uint16_t width, float_t* new_image, const uint16_t kernel_size) {

    uint16_t row, column;

    for(row = 0; row < height; row += kernel_size) {
        for(column = 0; column < width; column += kernel_size) {
            float_t pixel = original_image[row*width+column];
    
            uint16_t sub_row, sub_column;
            
            for(sub_row = row; sub_row < row+kernel_size; sub_row++) {
                for(sub_column = column; sub_column < column+kernel_size; sub_column++) {
                    if(original_image[sub_row*width+sub_column] > pixel) {
                        pixel = original_image[sub_row*width+sub_column];
                    }
                }
            }
            
            new_image[(row/kernel_size)*(height/kernel_size)+(column/kernel_size)] = pixel;
        }
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
void average_pooling2d_naive(const float_t* original_image, const uint16_t height, const uint16_t width, float_t* new_image, const uint16_t kernel_size, float_t bias) {

    uint16_t row, column;

    for(row = 0; row < height; row += kernel_size) {
        for(column = 0; column < width; column += kernel_size) {
            float_t pixel = original_image[row*width+column];
    
            uint16_t sub_row, sub_column;
            
            for(sub_row = row; sub_row < row+kernel_size; sub_row++) {
                for(sub_column = column; sub_column < column+kernel_size; sub_column++) {
                    pixel += original_image[sub_row*width+sub_column];
                }
            }
            
            new_image[(row/kernel_size)*(height/kernel_size)+(column/kernel_size)] = pixel/((float_t) kernel_size*kernel_size) + bias;
        }
    }
}
