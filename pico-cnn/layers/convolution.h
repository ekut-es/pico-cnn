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

    uint16_t image_row, image_column;
    uint8_t padding = 1;

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

    for(image_row = 0; image_row < height-padding; image_row++) {
        for(image_column = 0; image_column < width-padding-5; image_column+=5) {

            // load image into vectors
            uint32_t source_0 = (image_row+0)*width+image_column;
            uint32_t source_1 = (image_row+1)*width+image_column;
            uint32_t source_2 = (image_row+2)*width+image_column;

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
            uint32_t target = image_row*(width-2*padding)+image_column;

            new_image[target] =   image_0[0] + image_0[1] + image_0[2] + bias;
            new_image[target+1] = image_1[0] + image_1[1] + image_1[2] + bias;
            new_image[target+2] = image_2[0] + image_2[1] + image_2[2] + bias;
            new_image[target+3] = image_3[0] + image_3[1] + image_3[2] + bias;
            new_image[target+4] = image_4[0] + image_4[1] + image_4[2] + bias;
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
#endif

#endif // CONVOLUTION_H
