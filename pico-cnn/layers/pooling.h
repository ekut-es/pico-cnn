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

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif 

#ifdef ARM_NEON
#include "arm_neon.h"
#include <float.h>
#endif


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


#ifdef FIXED16
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
void max_pooling2d_naive_fixed16(const fixed16_t* original_image, const uint16_t height, const uint16_t width, fixed16_t* new_image, const uint16_t kernel_size, const uint16_t stride) {

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


void max_pooling2d_cpu_2x2_s2_fixed16(const fixed16_t* original_image, const uint16_t height, const uint16_t width, fixed16_t* new_image) {
    max_pooling2d_naive_fixed16(original_image, height, width, new_image, 2, 2);
}

#endif

#ifdef ARM_NEON 
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

    uint16_t stride = 2;

    uint16_t image_row, image_column;
    uint16_t new_image_row, new_image_column;
    uint16_t new_image_width;

    new_image_row = 0;
    new_image_column = 0;
    
    new_image_width = width/stride;


    float32x4_t original_image_0;
    float32x4_t original_image_1;
    float32x4_t original_image_2;
    float32x4_t original_image_3;
    float32x4_t original_image_4;
    float32x4_t original_image_5;
    float32x4_t original_image_6;
    float32x4_t original_image_7;

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


    for(image_row = 0; image_row < height-4; image_row += 4) {
        for(image_column = 0; image_column < width-8; image_column += 8) {
           
            const uint32_t source_0 = (image_row*width)+image_column;
            const uint32_t source_1 = ((image_row+1)*width)+image_column;
            const uint32_t source_2 = ((image_row+2)*width)+image_column;
            const uint32_t source_3 = ((image_row+3)*width)+image_column;

            // load image into vectors 
            original_image_0 = vld1q_f32(original_image+source_0);
            original_image_1 = vld1q_f32(original_image+source_1);
            original_image_2 = vld1q_f32(original_image+source_2);
            original_image_3 = vld1q_f32(original_image+source_3);

            original_image_4 = vld1q_f32(original_image+source_0+4);
            original_image_5 = vld1q_f32(original_image+source_1+4);
            original_image_6 = vld1q_f32(original_image+source_2+4);
            original_image_7 = vld1q_f32(original_image+source_3+4);

            // determine max of halfs
            temp_max_0 = vpmax_f32(vget_low_f32(original_image_0), vget_low_f32(original_image_1));
            temp_max_1 = vpmax_f32(vget_high_f32(original_image_0), vget_high_f32(original_image_1));
            temp_max_2 = vpmax_f32(vget_low_f32(original_image_2), vget_low_f32(original_image_3));
            temp_max_3 = vpmax_f32(vget_high_f32(original_image_2), vget_high_f32(original_image_3));

            temp_max_4 = vpmax_f32(vget_low_f32(original_image_4), vget_low_f32(original_image_5));
            temp_max_5 = vpmax_f32(vget_high_f32(original_image_4), vget_high_f32(original_image_5));
            temp_max_6 = vpmax_f32(vget_low_f32(original_image_6), vget_low_f32(original_image_7));
            temp_max_7 = vpmax_f32(vget_high_f32(original_image_6), vget_high_f32(original_image_7));


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

            const uint32_t target_0 = new_image_row*new_image_width+new_image_column;
            const uint32_t target_1 = (new_image_row+1)*new_image_width+new_image_column;

            new_image[target_0] = pixel_0;
            new_image[target_0+1] = pixel_1;
            new_image[target_1] = pixel_2;
            new_image[target_1+1] = pixel_3;

            new_image[target_0+2] = pixel_4;
            new_image[target_0+3] = pixel_5;
            new_image[target_1+2] = pixel_6;
            new_image[target_1+3] = pixel_7;

            new_image_column+=4;
        }

        // residual columns
        for(image_column = image_column; image_column < width; image_column+=2) {
            
            temp_max_0 = vld1_f32(original_image+(image_row*width)+image_column);
            temp_max_1 = vld1_f32(original_image+((image_row+1)*width)+image_column);
            temp_max_2 = vld1_f32(original_image+((image_row+2)*width)+image_column);
            temp_max_3 = vld1_f32(original_image+((image_row+3)*width)+image_column);

            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);
            temp_max_2 = vpmax_f32(temp_max_2, temp_max_2);
            temp_max_3 = vpmax_f32(temp_max_3, temp_max_3);

            pixel_0 = MAX(vget_lane_f32(temp_max_0, 0), vget_lane_f32(temp_max_1, 0));
            pixel_1 = MAX(vget_lane_f32(temp_max_2, 0), vget_lane_f32(temp_max_3, 0));

            new_image[new_image_row*new_image_width+new_image_column] = pixel_0;
            new_image[(new_image_row+1)*new_image_width+new_image_column] = pixel_1;

            new_image_column++;
        }

        new_image_row+=2;
        new_image_column = 0;
    }

    // residual rows
    for(image_row = image_row; image_row < height; image_row+=2) {
        for(image_column = 0; image_column < width-4; image_column += 4) {

            // load image into vectors 
            original_image_0 = vld1q_f32(original_image+(image_row*width)+image_column);
            original_image_1 = vld1q_f32(original_image+((image_row+1)*width)+image_column);

            // determine max of halfs
            temp_max_0 = vpmax_f32(vget_low_f32(original_image_0), vget_low_f32(original_image_1));
            temp_max_1 = vpmax_f32(vget_high_f32(original_image_0), vget_high_f32(original_image_1));

            // determine max of temp_max_*
            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);

            pixel_0 = vget_lane_f32(temp_max_0, 0);
            pixel_1 = vget_lane_f32(temp_max_1, 0);

            new_image[new_image_row*new_image_width+new_image_column] = pixel_0;
            new_image[new_image_row*new_image_width+new_image_column+1] = pixel_1;

            new_image_column+=2;
        }

        // residual columns
        for(image_column = image_column; image_column < width; image_column+=2) {

            temp_max_0 = vld1_f32(original_image+(image_row*width)+image_column);
            temp_max_1 = vld1_f32(original_image+((image_row+1)*width)+image_column);

            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);

            pixel_0 = MAX(vget_lane_f32(temp_max_0, 0), vget_lane_f32(temp_max_1, 0));

            new_image[new_image_row*new_image_width+new_image_column] = pixel_0;
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

    uint16_t stride = 2;

    uint16_t image_row, image_column;
    uint16_t new_image_row, new_image_column;
    uint16_t new_image_width;

    new_image_row = 0;
    new_image_column = 0;
    
    new_image_width = width/stride;

    float32x4_t original_image_0;
    float32x4_t original_image_1;
    float32x4_t original_image_2;

    float32x4_t original_image_3;
    float32x4_t original_image_4;
    float32x4_t original_image_5;

    float32x4_t original_image_6;
    float32x4_t original_image_7;
    float32x4_t original_image_8;

    float32x4_t original_image_9;
    float32x4_t original_image_10;
    float32x4_t original_image_11;


    float32x2_t temp_max_0;
    float32x2_t temp_max_1;
    float32x2_t temp_max_2;
    float32x2_t temp_max_3;

    fp_t pixel_0;
    fp_t pixel_1;
    fp_t pixel_2;
    fp_t pixel_3;

    for(image_row = 0; image_row < height-2; image_row += 2) {
        for(image_column = 0; image_column < width-8; image_column += 8) {

            const uint32_t source_0 = (image_row*width)+image_column;
            const uint32_t source_1 = ((image_row+1)*width)+image_column;
            const uint32_t source_2 = ((image_row+2)*width)+image_column;

            // load image into vectors 
            original_image_0 = vld1q_f32(original_image+source_0);
            original_image_1 = vld1q_f32(original_image+source_1);
            original_image_2 = vld1q_f32(original_image+source_2);

            original_image_3 = vld1q_f32(original_image+source_0+2);
            original_image_4 = vld1q_f32(original_image+source_1+2);
            original_image_5 = vld1q_f32(original_image+source_2+2);

            original_image_6 = vld1q_f32(original_image+source_0+4);
            original_image_7 = vld1q_f32(original_image+source_1+4);
            original_image_8 = vld1q_f32(original_image+source_2+4);

            original_image_9 =  vld1q_f32(original_image+source_0+6);
            original_image_10 = vld1q_f32(original_image+source_1+6);
            original_image_11 = vld1q_f32(original_image+source_2+6);


            // determine element wise max 
            original_image_0 = vmaxq_f32(original_image_0, original_image_1);
            original_image_0 = vmaxq_f32(original_image_0, original_image_2);

            original_image_3 = vmaxq_f32(original_image_3, original_image_4);
            original_image_3 = vmaxq_f32(original_image_3, original_image_5);

            original_image_6 = vmaxq_f32(original_image_6, original_image_7);
            original_image_6 = vmaxq_f32(original_image_6, original_image_8);

            original_image_9 = vmaxq_f32(original_image_9, original_image_10);
            original_image_9 = vmaxq_f32(original_image_9, original_image_11);


            // set last element to FLT_MIN
            original_image_0 = vsetq_lane_f32(-FLT_MAX, original_image_0, 3);
            original_image_3 = vsetq_lane_f32(-FLT_MAX, original_image_3, 3);
            original_image_6 = vsetq_lane_f32(-FLT_MAX, original_image_6, 3);
            original_image_9 = vsetq_lane_f32(-FLT_MAX, original_image_9, 3);

            // determine max
            temp_max_0 = vpmax_f32(vget_low_f32(original_image_0), vget_high_f32(original_image_0));
            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);

            temp_max_1 = vpmax_f32(vget_low_f32(original_image_3), vget_high_f32(original_image_3));
            temp_max_1 = vpmax_f32(temp_max_1, temp_max_1);

            temp_max_2 = vpmax_f32(vget_low_f32(original_image_6), vget_high_f32(original_image_6));
            temp_max_2 = vpmax_f32(temp_max_2, temp_max_2);

            temp_max_3 = vpmax_f32(vget_low_f32(original_image_9), vget_high_f32(original_image_9));
            temp_max_3 = vpmax_f32(temp_max_3, temp_max_3);


            pixel_0 = vget_lane_f32(temp_max_0, 0); 
            pixel_1 = vget_lane_f32(temp_max_1, 0); 
            pixel_2 = vget_lane_f32(temp_max_2, 0); 
            pixel_3 = vget_lane_f32(temp_max_3, 0); 

            const uint32_t target = new_image_row*new_image_width+new_image_column;

            new_image[target] = pixel_0;
            new_image[target+1] = pixel_1;
            new_image[target+2] = pixel_2;
            new_image[target+3] = pixel_3;

            new_image_column+=4;
        }

        // residual columns
        for(image_column = image_column; image_column < width; image_column+=2) {

            // load image into vectors 
            original_image_0 = vld1q_f32(original_image+(image_row*width)+image_column);
            original_image_1 = vld1q_f32(original_image+((image_row+1)*width)+image_column);
            original_image_2 = vld1q_f32(original_image+((image_row+2)*width)+image_column);

            // determine element wise max 
            original_image_0 = vmaxq_f32(original_image_0, original_image_1);
            original_image_0 = vmaxq_f32(original_image_0, original_image_2);

            // set last element to FLT_MIN
            original_image_0 = vsetq_lane_f32(FLT_MIN, original_image_0, 3);

            // determine max
            temp_max_0 = vpmax_f32(vget_low_f32(original_image_0), vget_high_f32(original_image_0));
            temp_max_0 = vpmax_f32(temp_max_0, temp_max_0);

            pixel_0 = vget_lane_f32(temp_max_0, 0); 
            new_image[new_image_row*new_image_width+new_image_column] = pixel_0;
            new_image_column++;
        }


        new_image_row++;
        new_image_column = 0;
    }
}
#endif

#endif // POOLING_H
