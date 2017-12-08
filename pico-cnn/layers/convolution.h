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

#include "../io/write_pgm.h"

#ifdef __aarch64__
#include "arm_neon.h"
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
    uint16_t row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            image_a[row*width+column] = (image_a[row*width+column] + image_b[row*width+column]);
        }
    }
}

#ifdef __aarch64__
/**
 * @brief performs an CPU optimized 2D convolution on original_image with a 
 * kernel 3x3 and stores the result to new_image
 *
 * stride = 1
 * padding = valid => image shrinks by 1 pixel
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height-1 x width-1)
 * @param kernel (3x3)
 * @param bias
 */
void convolution2d_cpu_3x3_s1_valid(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const fp_t bias) {

    /*
     3 float_32x4 neon registers for kernel (last value will be ignored)
     kernel_0 = [0,0][0,1][0,2][x]
     kernel_1 = [1,0][1,1][1,2][x]
     kernel_2 = [2,0][2,1][2,2][x]

     5x 3 float_32x4 neon registers for image (last value will be ignored)
     image_*_0 = [0,0][0,1][0,2][x]  [0,1][0,2][0,3][x]  [0,2][0,3][0,4][x]  [0,3][0,4][0,5][x]  [0,4][0,5][0,6][x]
     image_*_1 = [1,0][1,1][1,2][x]  [1,1][1,2][1,3][x]  [1,2][1,3][1,4][x]  [1,3][1,4][1,5][x]  [1,4][1,5][1,6][x]
     image_*_2 = [2,0][2,1][2,2][x]  [2,1][2,2][2,3][x]  [2,2][2,3][2,4][x]  [2,3][2,4][2,5][x]  [2,4][2,5][2,6][x]

     image_*_0 = image_*_0 * kernel_0
     image_*_1 = image_*_0 + image_*_1 * kernel_0
     image_*_2 = image_*_1 + image_*_2 * kernel_0
     */

    uint16_t image_row, image_column;
    const uint8_t padding = 1;

    // vectors for kernel
    float32x4_t kernel_0;
    float32x4_t kernel_1;
    float32x4_t kernel_2;

    kernel_0 = vld1q_f32(kernel);
    kernel_1 = vld1q_f32(kernel+3);
    kernel_2 = vld1q_f32(kernel+6);

    // vectors for image
    float32x4_t image_0_0;
    float32x4_t image_1_0;
    float32x4_t image_2_0;

    float32x4_t image_0_1;
    float32x4_t image_1_1;
    float32x4_t image_2_1;

    float32x4_t image_0_2;
    float32x4_t image_1_2;
    float32x4_t image_2_2;

    float32x4_t image_0_3;
    float32x4_t image_1_3;
    float32x4_t image_2_3;

    float32x4_t image_0_4;
    float32x4_t image_1_4;
    float32x4_t image_2_4;

    fp_t image_0[4];
    fp_t image_1[4];
    fp_t image_2[4];
    fp_t image_3[4];
    fp_t image_4[4];

    for(image_row = 0; image_row < height-2; image_row++) {
        for(image_column = 0; image_column < width-padding-5; image_column+=5) {

            // load image into vectors
            const uint32_t source_0 = (image_row+0)*width+image_column;
            const uint32_t source_1 = (image_row+1)*width+image_column;
            const uint32_t source_2 = (image_row+2)*width+image_column;

            image_0_0 = vld1q_f32(original_image+source_0);
            image_1_0 = vld1q_f32(original_image+source_1);
            image_2_0 = vld1q_f32(original_image+source_2);

            image_0_1 = vld1q_f32(original_image+source_0+1);
            image_1_1 = vld1q_f32(original_image+source_1+1);
            image_2_1 = vld1q_f32(original_image+source_2+1);

            image_0_2 = vld1q_f32(original_image+source_0+2);
            image_1_2 = vld1q_f32(original_image+source_1+2);
            image_2_2 = vld1q_f32(original_image+source_2+2);

            image_0_3 = vld1q_f32(original_image+source_0+3);
            image_1_3 = vld1q_f32(original_image+source_1+3);
            image_2_3 = vld1q_f32(original_image+source_2+3);

            image_0_4 = vld1q_f32(original_image+source_0+4);
            image_1_4 = vld1q_f32(original_image+source_1+4);
            image_2_4 = vld1q_f32(original_image+source_2+4);

            // apply kernel
            image_0_0 = vmulq_f32(image_0_0, kernel_0);
            image_1_0 = vmlaq_f32(image_0_0, image_1_0, kernel_1);
            image_2_0 = vmlaq_f32(image_1_0, image_2_0, kernel_2);

            image_0_1 = vmulq_f32(image_0_1, kernel_0);
            image_1_1 = vmlaq_f32(image_0_1, image_1_1, kernel_1);
            image_2_1 = vmlaq_f32(image_1_1, image_2_1, kernel_2);

            image_0_2 = vmulq_f32(image_0_2, kernel_0);
            image_1_2 = vmlaq_f32(image_0_2, image_1_2, kernel_1);
            image_2_2 = vmlaq_f32(image_1_2, image_2_2, kernel_2);
            
            image_0_3 = vmulq_f32(image_0_3, kernel_0);
            image_1_3 = vmlaq_f32(image_0_3, image_1_3, kernel_1);
            image_2_3 = vmlaq_f32(image_1_3, image_2_3, kernel_2);

            image_0_4 = vmulq_f32(image_0_4, kernel_0);
            image_1_4 = vmlaq_f32(image_0_4, image_1_4, kernel_1);
            image_2_4 = vmlaq_f32(image_1_4, image_2_4, kernel_2);

            // store vector into array
            vst1q_f32(image_0, image_2_0);
            vst1q_f32(image_1, image_2_1);
            vst1q_f32(image_2, image_2_2);
            vst1q_f32(image_3, image_2_3);
            vst1q_f32(image_4, image_2_4);

            // store new image
            const uint32_t target = image_row*(width-2*padding)+image_column;

            new_image[target] =   image_0[0] + image_0[1] + image_0[2] + bias;
            new_image[target+1] = image_1[0] + image_1[1] + image_1[2] + bias;
            new_image[target+2] = image_2[0] + image_2[1] + image_2[2] + bias;
            new_image[target+3] = image_3[0] + image_3[1] + image_3[2] + bias;
            new_image[target+4] = image_4[0] + image_4[1] + image_4[2] + bias;
        }

        // residual columns
        for(image_column = image_column; image_column < width-2; image_column++) {

            // load image into vectors
            const uint32_t source_0 = (image_row+0)*width+image_column;
            const uint32_t source_1 = (image_row+1)*width+image_column;
            const uint32_t source_2 = (image_row+2)*width+image_column;

            image_0_0 = vld1q_f32(original_image+source_0);
            image_1_0 = vld1q_f32(original_image+source_1);
            image_2_0 = vld1q_f32(original_image+source_2);

            // apply kernel
            image_0_0 = vmulq_f32(image_0_0, kernel_0);
            image_1_0 = vmlaq_f32(image_0_0, image_1_0, kernel_1);
            image_2_0 = vmlaq_f32(image_1_0, image_2_0, kernel_2);

            // store vector into array
            vst1q_f32(image_0, image_2_0);

            // store new image
            const uint32_t target = image_row*(width-2*padding)+image_column;

            new_image[target] = image_0[0] + image_0[1] + image_0[2] + bias;
        }
    }
}

/**
 * @brief performs an CPU optimized 2D convolution on original_image with a 
 * kernel 3x3 and stores the result to new_image
 *
 * stride = 1
 * padding = same
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 * @param kernel (3x3)
 * @param bias
 */
void convolution2d_cpu_3x3_s1_same(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const fp_t bias) {

    fp_t* bigger_original_image = (fp_t*) malloc((height+2)*(width+2)*sizeof(fp_t));
   
    // fill first line with 0.0
    int32_t i;
    for(i = 0; i < width+2; i++) {
        bigger_original_image[i] = 0.0;
    }
 
    int32_t image_row;
    for(image_row = 0; image_row < height; image_row++) {
        // set first pixel in row to 0.0
        bigger_original_image[(image_row+1)*(width+2)] = 0.0;

        memcpy(bigger_original_image+(image_row+1)*(width+2)+1, original_image+image_row*width, width*sizeof(fp_t));

        // set last pixel in row to 0.0
        bigger_original_image[(image_row+1)*(width+2)+(width+2)-1] = 0.0;
    }

    // fill last line with 0.0
    for(i = 0; i < width+2; i++) {
        bigger_original_image[(height+2-1)*(width+2)+i] = 0.0;
    }

    convolution2d_cpu_3x3_s1_valid(bigger_original_image, height+2, width+2, new_image, kernel, bias);
    free(bigger_original_image);
}

/**
 * @brief performs an CPU optimized 2D convolution on original_image with a 
 * kernel 5x5 and stores the result to new_image
 *
 * stride = 1
 * padding = valid => image shrinks by 2 pixels
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height-2 x width-2)
 * @param kernel (5x5)
 * @param bias
 */
void convolution2d_cpu_5x5_s1_valid(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const fp_t bias) {

    /*
    3 float_32x4 neon registers for kernel
                                    kernel_5 = 
    kernel_0 = [0,0][0,1][0,2][0,3]  [0,4]
    kernel_1 = [1,0][1,1][1,2][1,3]  [1,4]
    kernel_2 = [2,0][2,1][2,2][2,3]  [2,4]
    kernel_3 = [3,0][3,1][3,2][3,3]  [3,4]
    kernel_4 = [4,0][4,1][4,2][4,3]  [4,3] = kernel_6
    */

    uint16_t image_row, image_column;
    const uint8_t padding = 2;

    // vectors for kernel
    float32x4_t kernel_0;
    float32x4_t kernel_1;
    float32x4_t kernel_2;
    float32x4_t kernel_3;
    float32x4_t kernel_4;
    float32x4_t kernel_5;
    fp_t kernel_6;

    fp_t kernel_temp[4];

    kernel_0 = vld1q_f32(kernel);
    kernel_1 = vld1q_f32(kernel+5);
    kernel_2 = vld1q_f32(kernel+10);
    kernel_3 = vld1q_f32(kernel+15);
    kernel_4 = vld1q_f32(kernel+20);

    kernel_temp[0] = kernel[4];
    kernel_temp[1] = kernel[9];
    kernel_temp[2] = kernel[14];
    kernel_temp[3] = kernel[19];
    kernel_5 = vld1q_f32(kernel_temp);

    kernel_6 = kernel[24];

    // vectors for image
    float32x4_t image_0_0;
    float32x4_t image_1_0;
    float32x4_t image_2_0;
    float32x4_t image_3_0;
    float32x4_t image_4_0;
    float32x4_t image_5_0;
    fp_t image_6_0;

    float32x4_t image_0_1;
    float32x4_t image_1_1;
    float32x4_t image_2_1;
    float32x4_t image_3_1;
    float32x4_t image_4_1;
    float32x4_t image_5_1;
    fp_t image_6_1;

    float32x4_t image_0_2;
    float32x4_t image_1_2;
    float32x4_t image_2_2;
    float32x4_t image_3_2;
    float32x4_t image_4_2;
    float32x4_t image_5_2;
    fp_t image_6_2;

    float32x4_t image_0_3;
    float32x4_t image_1_3;
    float32x4_t image_2_3;
    float32x4_t image_3_3;
    float32x4_t image_4_3;
    float32x4_t image_5_3;
    fp_t image_6_3;

    fp_t image_0[4];
    fp_t image_1[4];
    fp_t image_2[4];
    fp_t image_3[4];

    fp_t image_temp[4];

    for(image_row = 0; image_row < height-4; image_row++) {
        for(image_column = 0; image_column < width-4-4; image_column+=4) {

            // load image into vectors
            const uint32_t source_0 = (image_row+0)*width+image_column;
            const uint32_t source_1 = (image_row+1)*width+image_column;
            const uint32_t source_2 = (image_row+2)*width+image_column;
            const uint32_t source_3 = (image_row+3)*width+image_column;
            const uint32_t source_4 = (image_row+4)*width+image_column;

            image_0_0 = vld1q_f32(original_image+source_0);
            image_1_0 = vld1q_f32(original_image+source_1);
            image_2_0 = vld1q_f32(original_image+source_2);
            image_3_0 = vld1q_f32(original_image+source_3);
            image_4_0 = vld1q_f32(original_image+source_4);

            image_temp[0] = original_image[source_0+4];
            image_temp[1] = original_image[source_1+4];
            image_temp[2] = original_image[source_2+4];
            image_temp[3] = original_image[source_3+4];
            image_5_0 = vld1q_f32(image_temp);

            image_6_0 = original_image[source_4+4]; 


            image_0_1 = vld1q_f32(original_image+source_0+1);
            image_1_1 = vld1q_f32(original_image+source_1+1);
            image_2_1 = vld1q_f32(original_image+source_2+1);
            image_3_1 = vld1q_f32(original_image+source_3+1);
            image_4_1 = vld1q_f32(original_image+source_4+1);

            image_temp[0] = original_image[source_0+4+1];
            image_temp[1] = original_image[source_1+4+1];
            image_temp[2] = original_image[source_2+4+1];
            image_temp[3] = original_image[source_3+4+1];
            image_5_1 = vld1q_f32(image_temp);

            image_6_1 = original_image[source_4+4+1];


            image_0_2 = vld1q_f32(original_image+source_0+2);
            image_1_2 = vld1q_f32(original_image+source_1+2);
            image_2_2 = vld1q_f32(original_image+source_2+2);
            image_3_2 = vld1q_f32(original_image+source_3+2);
            image_4_2 = vld1q_f32(original_image+source_4+2);

            image_temp[0] = original_image[source_0+4+2];
            image_temp[1] = original_image[source_1+4+2];
            image_temp[2] = original_image[source_2+4+2];
            image_temp[3] = original_image[source_3+4+2];
            image_5_2 = vld1q_f32(image_temp);

            image_6_2 = original_image[source_4+4+2];


            image_0_3 = vld1q_f32(original_image+source_0+3);
            image_1_3 = vld1q_f32(original_image+source_1+3);
            image_2_3 = vld1q_f32(original_image+source_2+3);
            image_3_3 = vld1q_f32(original_image+source_3+3);
            image_4_3 = vld1q_f32(original_image+source_4+3);

            image_temp[0] = original_image[source_0+4+3];
            image_temp[1] = original_image[source_1+4+3];
            image_temp[2] = original_image[source_2+4+3];
            image_temp[3] = original_image[source_3+4+3];
            image_5_3 = vld1q_f32(image_temp);

            image_6_3 = original_image[source_4+4+3];


            // apply kernel
            image_0_0 = vmulq_f32(image_0_0, kernel_0);
            image_1_0 = vmlaq_f32(image_0_0, image_1_0, kernel_1);
            image_2_0 = vmlaq_f32(image_1_0, image_2_0, kernel_2);
            image_3_0 = vmlaq_f32(image_2_0, image_3_0, kernel_3);
            image_4_0 = vmlaq_f32(image_3_0, image_4_0, kernel_4);
            image_5_0 = vmlaq_f32(image_4_0, image_5_0, kernel_5);

            image_0_1 = vmulq_f32(image_0_1, kernel_0);
            image_1_1 = vmlaq_f32(image_0_1, image_1_1, kernel_1);
            image_2_1 = vmlaq_f32(image_1_1, image_2_1, kernel_2);
            image_3_1 = vmlaq_f32(image_2_1, image_3_1, kernel_3);
            image_4_1 = vmlaq_f32(image_3_1, image_4_1, kernel_4);
            image_5_1 = vmlaq_f32(image_4_1, image_5_1, kernel_5);

            image_0_2 = vmulq_f32(image_0_2, kernel_0);
            image_1_2 = vmlaq_f32(image_0_2, image_1_2, kernel_1);
            image_2_2 = vmlaq_f32(image_1_2, image_2_2, kernel_2);
            image_3_2 = vmlaq_f32(image_2_2, image_3_2, kernel_3);
            image_4_2 = vmlaq_f32(image_3_2, image_4_2, kernel_4);
            image_5_2 = vmlaq_f32(image_4_2, image_5_2, kernel_5);

            image_0_3 = vmulq_f32(image_0_3, kernel_0);
            image_1_3 = vmlaq_f32(image_0_3, image_1_3, kernel_1);
            image_2_3 = vmlaq_f32(image_1_3, image_2_3, kernel_2);
            image_3_3 = vmlaq_f32(image_2_3, image_3_3, kernel_3);
            image_4_3 = vmlaq_f32(image_3_3, image_4_3, kernel_4);
            image_5_3 = vmlaq_f32(image_4_3, image_5_3, kernel_5);

            // store vector into array
            vst1q_f32(image_0, image_5_0);
            vst1q_f32(image_1, image_5_1);
            vst1q_f32(image_2, image_5_2);
            vst1q_f32(image_3, image_5_3);

            // store new image
            const uint32_t target = image_row*(width-2*padding)+image_column;

            new_image[target] =   image_0[0] + image_0[1] + image_0[2] + image_0[3] + (image_6_0*kernel_6) + bias;
            new_image[target+1] = image_1[0] + image_1[1] + image_1[2] + image_1[3] + (image_6_1*kernel_6) + bias;
            new_image[target+2] = image_2[0] + image_2[1] + image_2[2] + image_2[3] + (image_6_2*kernel_6) + bias;
            new_image[target+3] = image_3[0] + image_3[1] + image_3[2] + image_3[3] + (image_6_3*kernel_6) + bias;
        }

        // residual columns
        for(image_column = image_column; image_column < width-4; image_column++) {

            // load image into vectors
            const uint32_t source_0 = (image_row+0)*width+image_column;
            const uint32_t source_1 = (image_row+1)*width+image_column;
            const uint32_t source_2 = (image_row+2)*width+image_column;
            const uint32_t source_3 = (image_row+3)*width+image_column;
            const uint32_t source_4 = (image_row+4)*width+image_column;

            image_0_0 = vld1q_f32(original_image+source_0);
            image_1_0 = vld1q_f32(original_image+source_1);
            image_2_0 = vld1q_f32(original_image+source_2);
            image_3_0 = vld1q_f32(original_image+source_3);
            image_4_0 = vld1q_f32(original_image+source_4);

            image_temp[0] = original_image[source_0+4];
            image_temp[1] = original_image[source_1+4];
            image_temp[2] = original_image[source_2+4];
            image_temp[3] = original_image[source_3+4];
            image_5_0 = vld1q_f32(image_temp);

            image_6_0 = original_image[source_4+4];

            // apply kernel
            image_0_0 = vmulq_f32(image_0_0, kernel_0);
            image_1_0 = vmlaq_f32(image_0_0, image_1_0, kernel_1);
            image_2_0 = vmlaq_f32(image_1_0, image_2_0, kernel_2);
            image_3_0 = vmlaq_f32(image_2_0, image_3_0, kernel_3);
            image_4_0 = vmlaq_f32(image_3_0, image_4_0, kernel_4);
            image_5_0 = vmlaq_f32(image_4_0, image_5_0, kernel_5);

            // store vector into array
            vst1q_f32(image_0, image_5_0);

            // store new image
            const uint32_t target = image_row*(width-2*padding)+image_column;

            new_image[target] =   image_0[0] + image_0[1] + image_0[2] + image_0[3] + (image_6_0*kernel_6) + bias;
        }
    }
}

/**
 * @brief performs an CPU optimized 2D convolution on original_image with a 
 * kernel 3x3 and stores the result to new_image
 *
 * stride = 1
 * padding = same
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 * @param kernel (5x5)
 * @param bias
 */
void convolution2d_cpu_5x5_s1_same(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const fp_t bias) {

    fp_t* bigger_original_image = (fp_t*) malloc((height+4)*(width+4)*sizeof(fp_t));
   
    // fill first two lines with 0.0
    int32_t i;
    for(i = 0; i < (width+4)*2; i++) {
        bigger_original_image[i] = 0.0;
    }
 
    int32_t image_row;
    for(image_row = 0; image_row < height; image_row++) {
        // set first two pixels in row to 0.0
        bigger_original_image[(image_row+2)*(width+4)]   = 0.0;
        bigger_original_image[(image_row+2)*(width+4)+1] = 0.0;

        memcpy(bigger_original_image+(image_row+2)*(width+4)+2, original_image+image_row*width, width*sizeof(fp_t));

        // set last two pixels in row to 0.0
        bigger_original_image[(image_row+2)*(width+4)+(width+4)-2] = 0.0;
        bigger_original_image[(image_row+2)*(width+4)+(width+4)-1] = 0.0;
    }

    // fill last two lines with 0.0
    for(i = 0; i < width+4; i++) {
        bigger_original_image[(height+4-2)*(width+4)+i] = 0.0;
    }

    convolution2d_cpu_5x5_s1_valid(bigger_original_image, height+4, width+4, new_image, kernel, bias);
    free(bigger_original_image);
}

/**
 * @brief performs an CPU optimized 2D convolution on original_image with a 
 * kernel 11x11 and stores the result to new_image
 *
 * stride = 4
 * padding = valid
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image 
 * @param kernel (11x11)
 * @param bias
 */
void convolution2d_cpu_11x11_s4_valid(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image, const fp_t* kernel, const fp_t bias) {

    /*
    3 float_32x4 neon registers for kernel (last value will be ignored)
    kernel_0 =               kernel_1 =               kernel_2 =
    [0,0][0,1][0,2][0,3]/    [0,4][0,5][0,6][0,7]/    [0,8][0,9][0,10][x]/
    [5,0][5,1][5,2][5,3]/    [5,4][5,5][5,6][5,7]/    [5,8][5,9][5,10][x]/
    [10,0][10,1][10,2][10,3] [10,4][10,5][10,6][10,7] [10,8][10,9][10,10][x]

    kernel_3 =            kernel_4 =            kernel_5 =
    [1,0][1,1][1,2][1,3]/ [1,4][1,5][1,6][1,7]/ [1,8][1,9][1,10][x]/
    [6,0][6,1][6,2][6,3]  [6,4][6,5][6,6][6,7]  [6,8][6,9][6,10][x]
    kernel_6 =            kernel_7 =            kernel_8 =
    [2,0][2,1][2,2][2,3]/ [2,4][2,5][2,6][2,7]/ [2,8][2,9][2,10][x]/
    [7,0][7,1][7,2][7,3]  [7,4][7,5][7,6][7,7]  [7,8][7,9][7,10][x]
    kernel_9 =            kernel_10 =           kernel_11 =
    [3,0][3,1][3,2][3,3]  [3,4][3,5][3,6][3,7]  [3,8][3,9][3,10][x]
    [8,0][8,1][8,2][8,3]  [8,4][8,5][8,6][8,7]  [8,8][8,9][8,10][x]
    kernel_12 =           kernel_13 =           kernel_14 =
    [4,0][4,1][4,2][4,3]  [4,4][4,5][4,6][4,7]  [4,4][4,4][4,10][x]
    [9,0][9,1][9,2][9,3]  [9,4][9,5][9,6][9,7]  [9,8][9,9][9,10][x]
    */

    uint16_t image_row, image_column;
    uint16_t new_image_row, new_image_column;
    uint16_t new_image_width = ((width-2*5)/4)+1;

    // vectors for kernel
    float32x4_t kernel_0;
    float32x4_t kernel_1;
    float32x4_t kernel_2;
    float32x4_t kernel_3;
    float32x4_t kernel_4;
    float32x4_t kernel_5;
    float32x4_t kernel_6;
    float32x4_t kernel_7;
    float32x4_t kernel_8;
    float32x4_t kernel_9;
    float32x4_t kernel_10;
    float32x4_t kernel_11;
    float32x4_t kernel_12;
    float32x4_t kernel_13;
    float32x4_t kernel_14;

    kernel_0 = vld1q_f32(kernel);
    kernel_1 = vld1q_f32(kernel+5);
    kernel_2 = vld1q_f32(kernel+10);
    kernel_3 = vld1q_f32(kernel+15);
    kernel_4 = vld1q_f32(kernel+20);

    // vectors for image
    float32x4_t image_0_0;
    float32x4_t image_1_0;
    float32x4_t image_2_0;
    float32x4_t image_3_0;
    float32x4_t image_4_0;
    float32x4_t image_5_0;
    float32x4_t image_6_0;
    float32x4_t image_7_0;
    float32x4_t image_8_0;
    float32x4_t image_9_0;
    float32x4_t image_10_0;
    float32x4_t image_11_0;
    float32x4_t image_12_0;
    float32x4_t image_13_0;
    float32x4_t image_14_0;
    
    fp_t image_0[4];

    fp_t pixel;

    new_image_row = 0;
    new_image_column = 0;

    for(image_row = 0; image_row < height-10; image_row+=4) {
        for(image_column = 0; image_column < width-10; image_column+=4) {

            // load first part of kernel
            kernel_0 = vld1q_f32(kernel);
            kernel_1 = vld1q_f32(kernel+4);
            kernel_2 = vld1q_f32(kernel+8);

            kernel_3 = vld1q_f32(kernel+11);
            kernel_4 = vld1q_f32(kernel+15);
            kernel_5 = vld1q_f32(kernel+19);

            kernel_6 = vld1q_f32(kernel+22);
            kernel_7 = vld1q_f32(kernel+26);
            kernel_8 = vld1q_f32(kernel+30);

            kernel_9 = vld1q_f32(kernel+33);
            kernel_10 = vld1q_f32(kernel+37);
            kernel_11 = vld1q_f32(kernel+41);

            kernel_12 = vld1q_f32(kernel+44);
            kernel_13 = vld1q_f32(kernel+48);
            kernel_14 = vld1q_f32(kernel+52);

            // load first part of image
            image_0_0 = vld1q_f32(original_image+(image_row+0)*width+image_column);
            image_1_0 = vld1q_f32(original_image+(image_row+0)*width+image_column+4);
            image_2_0 = vld1q_f32(original_image+(image_row+0)*width+image_column+8);

            image_3_0 = vld1q_f32(original_image+(image_row+1)*width+image_column);
            image_4_0 = vld1q_f32(original_image+(image_row+1)*width+image_column+4);
            image_5_0 = vld1q_f32(original_image+(image_row+1)*width+image_column+8);

            image_6_0 = vld1q_f32(original_image+(image_row+2)*width+image_column);
            image_7_0 = vld1q_f32(original_image+(image_row+2)*width+image_column+4);
            image_8_0 = vld1q_f32(original_image+(image_row+2)*width+image_column+8);

            image_9_0  = vld1q_f32(original_image+(image_row+3)*width+image_column);
            image_10_0 = vld1q_f32(original_image+(image_row+3)*width+image_column+4);
            image_11_0 = vld1q_f32(original_image+(image_row+3)*width+image_column+8);

            image_12_0 = vld1q_f32(original_image+(image_row+4)*width+image_column);
            image_13_0 = vld1q_f32(original_image+(image_row+4)*width+image_column+4);
            image_14_0 = vld1q_f32(original_image+(image_row+4)*width+image_column+8);

            // apply kernel
            image_0_0 = vmulq_f32(image_0_0, kernel_0);
            image_1_0 = vmlaq_f32(image_0_0, image_1_0, kernel_1);

            image_3_0 = vmlaq_f32(image_1_0, image_3_0, kernel_3);
            image_4_0 = vmlaq_f32(image_3_0, image_4_0, kernel_4);

            image_6_0 = vmlaq_f32(image_4_0, image_6_0, kernel_6);
            image_7_0 = vmlaq_f32(image_6_0, image_7_0, kernel_7);

            image_9_0 = vmlaq_f32(image_7_0, image_9_0, kernel_9);
            image_10_0 = vmlaq_f32(image_9_0, image_10_0, kernel_10);

            image_12_0 = vmlaq_f32(image_10_0, image_12_0, kernel_12);
            image_13_0 = vmlaq_f32(image_12_0, image_13_0, kernel_13);
           
            vst1q_f32(image_0, image_13_0);

            pixel = image_0[0] + image_0[1] + image_0[2] + image_0[3];
            

            image_2_0 = vmulq_f32(image_2_0, kernel_2);
            image_5_0 = vmlaq_f32(image_2_0, image_5_0, kernel_5);
            image_8_0 = vmlaq_f32(image_5_0, image_8_0, kernel_8);
            image_11_0 = vmlaq_f32(image_8_0, image_11_0, kernel_11);
            image_14_0 = vmlaq_f32(image_11_0, image_14_0, kernel_14);
            
            vst1q_f32(image_0, image_14_0);

            pixel += image_0[0] + image_0[1] + image_0[2];


            // load second part of kernel
            kernel_0 = vld1q_f32(kernel+55);
            kernel_1 = vld1q_f32(kernel+59);
            kernel_2 = vld1q_f32(kernel+63);

            kernel_3 = vld1q_f32(kernel+66);
            kernel_4 = vld1q_f32(kernel+70);
            kernel_5 = vld1q_f32(kernel+74);

            kernel_6 = vld1q_f32(kernel+77);
            kernel_7 = vld1q_f32(kernel+81);
            kernel_8 = vld1q_f32(kernel+85);

            kernel_9 = vld1q_f32(kernel+88);
            kernel_10 = vld1q_f32(kernel+92);
            kernel_11 = vld1q_f32(kernel+96);

            kernel_12 = vld1q_f32(kernel+99);
            kernel_13 = vld1q_f32(kernel+103);
            kernel_14 = vld1q_f32(kernel+107);

            // load second part of image
            image_0_0 = vld1q_f32(original_image+(image_row+5)*width+image_column);
            image_1_0 = vld1q_f32(original_image+(image_row+5)*width+image_column+4);
            image_2_0 = vld1q_f32(original_image+(image_row+5)*width+image_column+8);

            image_3_0 = vld1q_f32(original_image+(image_row+6)*width+image_column);
            image_4_0 = vld1q_f32(original_image+(image_row+6)*width+image_column+4);
            image_5_0 = vld1q_f32(original_image+(image_row+6)*width+image_column+8);

            image_6_0 = vld1q_f32(original_image+(image_row+7)*width+image_column);
            image_7_0 = vld1q_f32(original_image+(image_row+7)*width+image_column+4);
            image_8_0 = vld1q_f32(original_image+(image_row+7)*width+image_column+8);

            image_9_0  = vld1q_f32(original_image+(image_row+8)*width+image_column);
            image_10_0 = vld1q_f32(original_image+(image_row+8)*width+image_column+4);
            image_11_0 = vld1q_f32(original_image+(image_row+8)*width+image_column+8);

            image_12_0 = vld1q_f32(original_image+(image_row+9)*width+image_column);
            image_13_0 = vld1q_f32(original_image+(image_row+9)*width+image_column+4);
            image_14_0 = vld1q_f32(original_image+(image_row+9)*width+image_column+8);

            // apply kernel
            image_0_0 = vmulq_f32(image_0_0, kernel_0);
            image_1_0 = vmlaq_f32(image_0_0, image_1_0, kernel_1);

            image_3_0 = vmlaq_f32(image_1_0, image_3_0, kernel_3);
            image_4_0 = vmlaq_f32(image_3_0, image_4_0, kernel_4);

            image_6_0 = vmlaq_f32(image_4_0, image_6_0, kernel_6);
            image_7_0 = vmlaq_f32(image_6_0, image_7_0, kernel_7);

            image_9_0 = vmlaq_f32(image_7_0, image_9_0, kernel_9);
            image_10_0 = vmlaq_f32(image_9_0, image_10_0, kernel_10);

            image_12_0 = vmlaq_f32(image_10_0, image_12_0, kernel_12);
            image_13_0 = vmlaq_f32(image_12_0, image_13_0, kernel_13);
           
            vst1q_f32(image_0, image_13_0);

            pixel += image_0[0] + image_0[1] + image_0[2] + image_0[3];
            

            image_2_0 = vmulq_f32(image_2_0, kernel_2);
            image_5_0 = vmlaq_f32(image_2_0, image_5_0, kernel_5);
            image_8_0 = vmlaq_f32(image_5_0, image_8_0, kernel_8);
            image_11_0 = vmlaq_f32(image_8_0, image_11_0, kernel_11);
            image_14_0 = vmlaq_f32(image_11_0, image_14_0, kernel_14);
            
            vst1q_f32(image_0, image_14_0);

            pixel += image_0[0] + image_0[1] + image_0[2];


            // load third part of kernel
            kernel_0 = vld1q_f32(kernel+110);
            kernel_1 = vld1q_f32(kernel+114);
            kernel_2 = vld1q_f32(kernel+118);

            // load third part of image
            image_0_0 = vld1q_f32(original_image+(image_row+10)*width+image_column);
            image_1_0 = vld1q_f32(original_image+(image_row+10)*width+image_column+4);
            image_2_0 = vld1q_f32(original_image+(image_row+10)*width+image_column+8);

            // apply kernel
            image_0_0 = vmulq_f32(image_0_0, kernel_0);
            image_1_0 = vmlaq_f32(image_0_0, image_1_0, kernel_1);

            vst1q_f32(image_0, image_1_0);

            pixel += image_0[0] + image_0[1] + image_0[2] + image_0[3];

            image_2_0 = vmulq_f32(image_2_0, kernel_2);

            vst1q_f32(image_0, image_2_0);

            pixel += image_0[0] + image_0[1] + image_0[2] + bias;

            // store new image
            new_image[new_image_row*new_image_width+new_image_column] = pixel;
            new_image_column++;
        }
        new_image_row++;
        new_image_column = 0;
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
#endif

#endif // CONVOLUTION_H
