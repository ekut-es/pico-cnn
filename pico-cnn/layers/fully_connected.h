/** 
 * @brief contains the fully connected layer
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include "../parameters.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __aarch64__
#include "arm_neon.h"
#endif

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif 



/**
 * @brief implementation of fully connected layer
 *
 * @param original_image (1 x original_width)
 * @param original_width
 * @param new_image (1 x new_width)
 * @param new_width
 * @param kernel
 * @param bias
 */
void fully_connected_naive(const fp_t* original_image, const uint16_t original_width, fp_t* new_image, const uint16_t new_width, const fp_t* kernel, const fp_t* bias) {

    int i, j;
    for(i = 0; i < new_width; i++) {

        fp_t pixel = 0.0;

        for(j = 0; j < original_width; j++) {
            // takes each new_width'nd element
            pixel += original_image[j] * kernel[j*new_width+i];
        }

        pixel += bias[i];
        new_image[i] = pixel;
    }
}

#ifdef __aarch64__
/**
 * @brief resturctes kernel for fully connected layer such that a vectorized
 * access is possible
 *
 * @param original_kernel
 * @param input_width of fully connected layer
 * @param output_width of fully connected layer
 */
void restructure_fully_connected_kernel(fp_t** original_kernel, const uint16_t input_width, const uint16_t output_width) {
    fp_t* new_kernel = (fp_t*) malloc(input_width*output_width*sizeof(fp_t));

    uint32_t i,j;

    for(i = 0 ; i < output_width; i++) {
        for(j = 0; j < input_width; j++) {
            new_kernel[i*input_width+j] = (*original_kernel)[j*output_width+i];
        }
    }

    //memcpy((*original_kernel), new_kernel, input_width*output_width*sizeof(fp_t));
    //free(new_kernel);
    free(*original_kernel);
    *original_kernel = new_kernel;
}

/**
 * @brief implementation of fully connected layer optimzed for CPU
 *
 * @param original_image (1 x original_width)
 * @param original_width
 * @param new_image (1 x new_width)
 * @param new_width
 * @param kernel
 * @param bias
 */
void fully_connected_cpu(const fp_t* original_image, const uint16_t original_width, fp_t* new_image, const uint16_t new_width, const fp_t* kernel, const fp_t* bias, const uint32_t from, const uint32_t to) {

    float32x4_t kernel_0 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t kernel_1 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t kernel_2 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t kernel_3 = {0.0, 0.0, 0.0, 0.0};

    float32x4_t original_image_0;
    float32x4_t original_image_1;
    float32x4_t original_image_2;
    float32x4_t original_image_3;


    uint32_t i, j;
    for(i = from; i < to; i++) {

        fp_t pixel = 0.0;

		// L1 block size is 64 Bytes = 16 floats
        for(j = 0; j < original_width-16; j+=16) {
            // load kernel into vectors
            /*
            kernel_0 = vsetq_lane_f32(kernel[j*new_width+i], kernel_0, 0);
            kernel_0 = vsetq_lane_f32(kernel[(j+1)*new_width+i], kernel_0, 1);
            kernel_0 = vsetq_lane_f32(kernel[(j+2)*new_width+i], kernel_0, 2);
            kernel_0 = vsetq_lane_f32(kernel[(j+3)*new_width+i], kernel_0, 3);

            kernel_1 = vsetq_lane_f32(kernel[(j+4)*new_width+i], kernel_1, 0);
            kernel_1 = vsetq_lane_f32(kernel[(j+5)*new_width+i], kernel_1, 1);
            kernel_1 = vsetq_lane_f32(kernel[(j+6)*new_width+i], kernel_1, 2);
            kernel_1 = vsetq_lane_f32(kernel[(j+7)*new_width+i], kernel_1, 3);

            kernel_2 = vsetq_lane_f32(kernel[(j+8)*new_width+i], kernel_2, 0);
            kernel_2 = vsetq_lane_f32(kernel[(j+9)*new_width+i], kernel_2, 1);
            kernel_2 = vsetq_lane_f32(kernel[(j+10)*new_width+i], kernel_2, 2);
            kernel_2 = vsetq_lane_f32(kernel[(j+11)*new_width+i], kernel_2, 3);

            kernel_3 = vsetq_lane_f32(kernel[(j+12)*new_width+i], kernel_3, 0);
            kernel_3 = vsetq_lane_f32(kernel[(j+13)*new_width+i], kernel_3, 1);
            kernel_3 = vsetq_lane_f32(kernel[(j+14)*new_width+i], kernel_3, 2);
            kernel_3 = vsetq_lane_f32(kernel[(j+15)*new_width+i], kernel_3, 3);
            */

            kernel_0 = vld1q_f32(kernel+i*original_width+j);
            kernel_1 = vld1q_f32(kernel+i*original_width+j+4);
            kernel_2 = vld1q_f32(kernel+i*original_width+j+8);
            kernel_3 = vld1q_f32(kernel+i*original_width+j+12);

            // load image into vectors
            original_image_0 = vld1q_f32(original_image+j);
            original_image_1 = vld1q_f32(original_image+j+4);
            original_image_2 = vld1q_f32(original_image+j+8);
            original_image_3 = vld1q_f32(original_image+j+12);

            // apply kernel
            original_image_0 = vmulq_f32(original_image_0, kernel_0);
            original_image_1 = vmulq_f32(original_image_1, kernel_1);
            original_image_2 = vmulq_f32(original_image_2, kernel_2);
            original_image_3 = vmulq_f32(original_image_3, kernel_3);

            // sum up 
            original_image_0 = vaddq_f32(original_image_0, original_image_1);
            original_image_0 = vaddq_f32(original_image_0, original_image_2);
            original_image_0 = vaddq_f32(original_image_0, original_image_3);

            // store in pixel
            pixel += vaddvq_f32(original_image_0);
        }

        // residual pixels
        for(j = j; j < original_width; j++) {
            //pixel += original_image[j] * kernel[j*new_width+i];
            pixel += original_image[j] * kernel[i*original_width+j];
        }

        pixel += bias[i];
        new_image[i] = pixel;
    }
}
#endif

#ifdef FIXED16
/**
 * @brief implementation of fully connected layer
 *
 * @param original_image (1 x original_width)
 * @param original_width
 * @param new_image (1 x new_width)
 * @param new_width
 * @param kernel
 * @param bias
 */
void fully_connected_naive_fixed16(const fixed16_t* original_image, const uint16_t original_width, fixed16_t* new_image, const uint16_t new_width, const fixed16_t* kernel, const fixed16_t* bias) {

    int i, j;
    for(i = 0; i < new_width; i++) {

        fp_t pixel = 0.0;

        for(j = 0; j < original_width; j++) {
            // takes each new_width'nd element
            pixel = add_fixed16(pixel, mul_fixed16(original_image[j], kernel[j*new_width+i]));
        }

        pixel = add_fixed16(pixel, bias[i]);
        new_image[i] = pixel;
    }
}

void fully_connected_cpu_fixed16(const fixed16_t* original_image, const uint16_t original_width, fixed16_t* new_image, const uint16_t new_width, const fixed16_t* kernel, const fixed16_t* bias, const uint32_t from, const uint32_t to) {
	fixed16x4_t kernel_0 = {0x0000, 0x0000, 0x0000, 0x0000};
    fixed16x4_t kernel_1 = {0x0000, 0x0000, 0x0000, 0x0000};
    fixed16x4_t kernel_2 = {0x0000, 0x0000, 0x0000, 0x0000};
    fixed16x4_t kernel_3 = {0x0000, 0x0000, 0x0000, 0x0000};
    fixed16x4_t kernel_4 = {0x0000, 0x0000, 0x0000, 0x0000};
    fixed16x4_t kernel_5 = {0x0000, 0x0000, 0x0000, 0x0000};
    fixed16x4_t kernel_6 = {0x0000, 0x0000, 0x0000, 0x0000};
    fixed16x4_t kernel_7 = {0x0000, 0x0000, 0x0000, 0x0000};

    fixed16x4_t original_image_0;
    fixed16x4_t original_image_1;
    fixed16x4_t original_image_2;
    fixed16x4_t original_image_3;
    fixed16x4_t original_image_4;
    fixed16x4_t original_image_5;
    fixed16x4_t original_image_6;
    fixed16x4_t original_image_7;


    uint32_t i, j;
    for(i = from; i < to; i++) {

        fp_t pixel = 0.0;

		// L1 block size is 64 Bytes = 32 fixed16_t
        for(j = 0; j < original_width-32; j+=32) {
            // load kernel into vectors
			/*
            kernel_0 = vset_lane_s16(kernel[j*new_width+i], kernel_0, 0);
            kernel_0 = vset_lane_s16(kernel[(j+1)*new_width+i], kernel_0, 1);
            kernel_0 = vset_lane_s16(kernel[(j+2)*new_width+i], kernel_0, 2);
            kernel_0 = vset_lane_s16(kernel[(j+3)*new_width+i], kernel_0, 3);

            kernel_1 = vset_lane_s16(kernel[(j+4)*new_width+i], kernel_1, 0);
            kernel_1 = vset_lane_s16(kernel[(j+5)*new_width+i], kernel_1, 1);
            kernel_1 = vset_lane_s16(kernel[(j+6)*new_width+i], kernel_1, 2);
            kernel_1 = vset_lane_s16(kernel[(j+7)*new_width+i], kernel_1, 3);

            kernel_2 = vset_lane_s16(kernel[(j+8)*new_width+i], kernel_2, 0);
            kernel_2 = vset_lane_s16(kernel[(j+9)*new_width+i], kernel_2, 1);
            kernel_2 = vset_lane_s16(kernel[(j+10)*new_width+i], kernel_2, 2);
            kernel_2 = vset_lane_s16(kernel[(j+11)*new_width+i], kernel_2, 3);

            kernel_3 = vset_lane_s16(kernel[(j+12)*new_width+i], kernel_3, 0);
            kernel_3 = vset_lane_s16(kernel[(j+13)*new_width+i], kernel_3, 1);
            kernel_3 = vset_lane_s16(kernel[(j+14)*new_width+i], kernel_3, 2);
            kernel_3 = vset_lane_s16(kernel[(j+15)*new_width+i], kernel_3, 3);
			*/

            kernel_0 = vld1_s16(kernel+i*original_width+j);
            kernel_1 = vld1_s16(kernel+i*original_width+j+4);
            kernel_2 = vld1_s16(kernel+i*original_width+j+8);
            kernel_3 = vld1_s16(kernel+i*original_width+j+12);
            kernel_4 = vld1_s16(kernel+i*original_width+j+16);
            kernel_5 = vld1_s16(kernel+i*original_width+j+20);
            kernel_6 = vld1_s16(kernel+i*original_width+j+24);
            kernel_7 = vld1_s16(kernel+i*original_width+j+28);

            // load image into vectors
            original_image_0 = vld1_s16(original_image+j);
            original_image_1 = vld1_s16(original_image+j+4);
            original_image_2 = vld1_s16(original_image+j+8);
            original_image_3 = vld1_s16(original_image+j+12);
            original_image_4 = vld1_s16(original_image+j+16);
            original_image_5 = vld1_s16(original_image+j+20);
            original_image_6 = vld1_s16(original_image+j+24);
            original_image_7 = vld1_s16(original_image+j+28);

            // apply kernel
            original_image_0 = vmul_fixed16(original_image_0, kernel_0);
            original_image_1 = vmul_fixed16(original_image_1, kernel_1);
            original_image_2 = vmul_fixed16(original_image_2, kernel_2);
            original_image_3 = vmul_fixed16(original_image_3, kernel_3);
            original_image_4 = vmul_fixed16(original_image_4, kernel_4);
            original_image_5 = vmul_fixed16(original_image_5, kernel_5);
            original_image_6 = vmul_fixed16(original_image_6, kernel_6);
            original_image_7 = vmul_fixed16(original_image_7, kernel_7);

            // sum up 
            original_image_0 = vadd_s16(original_image_0, original_image_1);
            original_image_0 = vadd_s16(original_image_0, original_image_2);
            original_image_0 = vadd_s16(original_image_0, original_image_3);
            original_image_0 = vadd_s16(original_image_0, original_image_4);
            original_image_0 = vadd_s16(original_image_0, original_image_5);
            original_image_0 = vadd_s16(original_image_0, original_image_6);
            original_image_0 = vadd_s16(original_image_0, original_image_7);

            // store in pixel
            pixel += vaddv_s16(original_image_0);
        }

        // residual pixels
        for(j = j; j < original_width; j++) {
            pixel = add_fixed16(pixel, mul_fixed16(original_image[j], kernel[i*original_width+j]));
        }

        pixel = add_fixed16(pixel, bias[i]);
        new_image[i] = pixel;
    }

}

#endif

#endif // FULLY_CONNECTED_H
