/** 
 * @brief contains all activation functions 
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#ifdef FIXED16
#include "../driver/fixed16.h"
#endif 

#ifdef ARM_NEON
#include "arm_neon.h"
#include "../driver/neon_mathfun.h"
#endif


/**
 * @brief applies tanh(x) to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void tanh_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    for(i = 0; i < height*width; i++) {
        new_image[i] = tanhf(original_image[i]);
    }
}

/**
 * @brief applies relu(x) to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void relu_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    for(i = 0; i < height*width; i++) {
        new_image[i] = (original_image[i] < 0.0) ? 0.0 : original_image[i];
    }
}

/**
 * @brief applies softmax to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void softmax_naive(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    fp_t denominator = 0.0;

    for(i = 0; i < height*width; i++) {
        denominator += expf(original_image[i]);
    }

    for(i = 0; i < height*width; i++) {
        new_image[i] = expf(original_image[i]) / denominator;
    }
}

/**
 * @brief performs a local response normalization (across channels) on original 
 * image and stores the result in new_image
 *
 * Formula (Paper):
 * https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn/252343#252343
 * Formula (Implemented):
 * http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param depth
 * @param new_image
 * @param alpha
 * @param beta
 * @param n
 */
void local_response_normalization_naive(fp_t** original_image, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** new_image, const fp_t alpha, const fp_t beta, const uint16_t n) {
    
    int32_t channel, row, column, i;
    int32_t from;
    int32_t to;

    fp_t sum;

    for(channel = 0; channel < depth; channel++) {
        from = MAX(0,channel-(n/2));
        to = MIN(depth-1,channel+(n/2));

        for(row = 0; row < height; row++) {
            for(column = 0; column < width; column++) {

                sum = 0.0;

                for(i = from; i <= to; i++) {
                    sum += powf(original_image[i][row*width+column], 2);
                }

                new_image[channel][row*width+column] = original_image[channel][row*width+column] / powf((1+(alpha/n)*sum),beta);
            }
        }
    }
}


#ifdef FIXED16
/**
 * @brief applies relu(x) to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void relu_naive_fixed16(const fixed16_t* original_image, const uint16_t height, const uint16_t width, fixed16_t* new_image) {

    uint16_t i;

    for(i = 0; i < height*width; i++) {
        new_image[i] = ((original_image[i] & 0x8000) == 0x8000) ? 0 : original_image[i];
    }
}

void relu_cpu_fixed16(const fixed16_t* original_image, const uint16_t height, const uint16_t width, fixed16_t* new_image) {
    relu_naive_fixed16(original_image, height, width, new_image);
}

/**
 * @brief applies softmax to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void softmax_naive_fixed16(const fixed16_t* original_image, const uint16_t height, const uint16_t width, fixed16_t* new_image) {

    uint16_t i;

    fixed16_t denominator = FIXED_ZERO;

    for(i = 0; i < height*width; i++) {
        denominator += exp_int32(fixed16_to_int16(original_image[i]));
    }

    for(i = 0; i < height*width; i++) {
        new_image[i] = div_fixed16(exp_int32(fixed16_to_int16(original_image[i])), denominator);
    }

}
#endif

#ifdef ARM_NEON 
/**
 * @brief applies relu(x) to all pixel of original_image and stores it in
 * new_image optimzed of CPU
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void relu_cpu(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    float32x4_t original_image_0;
    float32x4_t original_image_1;
    float32x4_t original_image_2;
    float32x4_t original_image_3;

    float32x4_t new_image_0;
    float32x4_t new_image_1;
    float32x4_t new_image_2;
    float32x4_t new_image_3;

    float32x4_t zero = {0.0, 0.0, 0.0, 0.0};

    uint32_t i;

    for(i = 0; i < height*width-BLOCK_SIZE; i += BLOCK_SIZE) {

        // load image into vectors
        original_image_0 = vld1q_f32(original_image+i);
        original_image_1 = vld1q_f32(original_image+i+4);
        original_image_2 = vld1q_f32(original_image+i+8);
        original_image_3 = vld1q_f32(original_image+i+12);

        new_image_0 = vmaxq_f32(original_image_0, zero);
        new_image_1 = vmaxq_f32(original_image_1, zero);
        new_image_2 = vmaxq_f32(original_image_2, zero);
        new_image_3 = vmaxq_f32(original_image_3, zero);

        vst1q_f32(new_image+i, new_image_0);
        vst1q_f32(new_image+i+4, new_image_1);
        vst1q_f32(new_image+i+8, new_image_2);
        vst1q_f32(new_image+i+12, new_image_3);
    }

    // residual pixels
    for(i = i; i < height*width; i++) {
        new_image[i] = (original_image[i] < 0.0) ? 0.0 : original_image[i];
    }
}

/**
 * @brief applies softmax to all pixel of original_image and stores it in
 * new_image optimzed of CPU
 * Only single core optimization since softmax is usually performed on a small
 * dataset and a multi core solution would impose a large overhead
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void softmax_cpu_single(const fp_t* original_image, const uint16_t height, const uint16_t width, fp_t* new_image) {

    uint16_t i;

    fp_t denominator = 0.0;

    float32x4_t original_image_0;
    float32x4_t original_image_1;
    float32x4_t original_image_2;
    float32x4_t original_image_3;

    // calculate denominator
    for(i = 0; i < height*width-BLOCK_SIZE; i += BLOCK_SIZE) {
        // load image into vectors
        original_image_0 = vld1q_f32(original_image+i);
        original_image_1 = vld1q_f32(original_image+i+4);
        original_image_2 = vld1q_f32(original_image+i+8);
        original_image_3 = vld1q_f32(original_image+i+12);

        // apply exponential function to vectors
        original_image_0 = exp_ps(original_image_0);
        original_image_1 = exp_ps(original_image_1);
        original_image_2 = exp_ps(original_image_2);
        original_image_3 = exp_ps(original_image_3);

        // add vectors together
        original_image_1 = vaddq_f32(original_image_1, original_image_0);
        original_image_2 = vaddq_f32(original_image_2, original_image_1);
        original_image_3 = vaddq_f32(original_image_3, original_image_2);

        // sum up whole vector
        denominator += vgetq_lane_f32(original_image_3, 0) + vgetq_lane_f32(original_image_3, 1) + vgetq_lane_f32(original_image_3, 2) + vgetq_lane_f32(original_image_3, 3); //vaddvq_f32(original_image_3);
    }

    // residual pixels
    for(i = i; i < height*width; i++) {
        denominator += expf(original_image[i]);
    }
   
    const fp_t inv_denominator = 1.0/denominator;
    // apply softmax
    for(i = 0; i < height*width-BLOCK_SIZE; i += BLOCK_SIZE) {
        // load image into vectors
        original_image_0 = vld1q_f32(original_image+i);
        original_image_1 = vld1q_f32(original_image+i+4);
        original_image_2 = vld1q_f32(original_image+i+8);
        original_image_3 = vld1q_f32(original_image+i+12);

        // apply exponential function to vectors 
        original_image_0 = exp_ps(original_image_0);
        original_image_1 = exp_ps(original_image_1);
        original_image_2 = exp_ps(original_image_2);
        original_image_3 = exp_ps(original_image_3);

        // multiply vectors scalar with inverted denominator
        original_image_0 = vmulq_n_f32(original_image_0, inv_denominator);
        original_image_1 = vmulq_n_f32(original_image_1, inv_denominator);
        original_image_2 = vmulq_n_f32(original_image_2, inv_denominator);
        original_image_3 = vmulq_n_f32(original_image_3, inv_denominator);

        // store vectors in new image
        vst1q_f32(new_image+i, original_image_0);
        vst1q_f32(new_image+i+4, original_image_1);
        vst1q_f32(new_image+i+8, original_image_2);
        vst1q_f32(new_image+i+12, original_image_3);
    }

    // residual pixels
    for(i = i; i < height*width; i++) {
        new_image[i] = expf(original_image[i]) * inv_denominator;
    }
}

/**
 * @brief performs a local response normalization (across channels) on original 
 * image and stores the result in new_image optimized for single CPU
 *
 * Formula (Paper):
 * https://stats.stackexchange.com/questions/145768/importance-of-local-response-normalization-in-cnn/252343#252343
 * Formula (Implemented):
 * http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param depth
 * @param new_image
 * @param alpha
 * @param beta
 * @param n
 */
void local_response_normalization_cpu_single(fp_t** original_image, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** new_image, const fp_t alpha, const fp_t beta, const uint16_t n) {
    int32_t channel, row, column, i;
    int32_t from;
    int32_t to;

    fp_t sum_0;
    fp_t sum_1;
    fp_t sum_2;
    fp_t sum_3;

    float32x4_t denominator_0;
    float32x4_t sums_0 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t original_image_0 = {0.0, 0.0, 0.0, 0.0};
    float32x4_t new_image_0;

    float32x4_t one = {1.0, 1.0, 1.0, 1.0};

    fp_t denominator_temp[4];
    fp_t new_image_temp[4];

    for(channel = 0; channel < depth; channel++) {
        from = MAX(0,channel-(n/2));
        to = MIN(depth-1,channel+(n/2));

        for(row = 0; row < height; row++) {
            for(column = 0; column < width-4; column+=4) {

                sum_0 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_0 += original_image[i][row*width+column]*original_image[i][row*width+column];
                }

                sum_1 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_1 += original_image[i][row*width+column+1]*original_image[i][row*width+column+1];
                }

                sum_2 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_2 += original_image[i][row*width+column+2]*original_image[i][row*width+column+2];
                }

                sum_3 = 0.0;
                for(i = from; i <= to; i++) {
                    sum_3 += original_image[i][row*width+column+3]*original_image[i][row*width+column+3];
                }


                const fp_t alpha_n = alpha/n;

                // load vector with sums
                sums_0 = vsetq_lane_f32(sum_0, sums_0, 0);
                sums_0 = vsetq_lane_f32(sum_1, sums_0, 1);
                sums_0 = vsetq_lane_f32(sum_2, sums_0, 2);
                sums_0 = vsetq_lane_f32(sum_3, sums_0, 3);

                // sums multiply with alpha/n
                denominator_0 = vmulq_n_f32(sums_0, alpha_n);
                // add 1
                denominator_0 = vaddq_f32(denominator_0, one);
                // store denominator vector in array
                vst1q_f32(denominator_temp, denominator_0);

                denominator_temp[0] = powf(denominator_temp[0],beta);
                denominator_temp[1] = powf(denominator_temp[1],beta);
                denominator_temp[2] = powf(denominator_temp[2],beta);
                denominator_temp[3] = powf(denominator_temp[3],beta);

                // load array back into vector
                denominator_0 = vld1q_f32(denominator_temp);

                // denominator = 1/denomniator
                denominator_0 = vrecpeq_f32(denominator_0);

                // store denominator vector in array
                vst1q_f32(denominator_temp, denominator_0);

                // load original image into vector
                original_image_0 = vsetq_lane_f32(original_image[channel][row*width+column],   original_image_0, 0);
                original_image_0 = vsetq_lane_f32(original_image[channel][row*width+column+1], original_image_0, 1);
                original_image_0 = vsetq_lane_f32(original_image[channel][row*width+column+2], original_image_0, 2);
                original_image_0 = vsetq_lane_f32(original_image[channel][row*width+column+3], original_image_0, 3);

                // new_image = original_image * (1/denominator)
                new_image_0 = vmulq_f32(original_image_0, denominator_0);

                // store new_image vector into array
                vst1q_f32(new_image_temp, new_image_0);

                new_image[channel][row*width+column] =   new_image_temp[0]; 
                new_image[channel][row*width+column+1] = new_image_temp[1];
                new_image[channel][row*width+column+2] = new_image_temp[2];
                new_image[channel][row*width+column+3] = new_image_temp[3];
            }

            // residual columns
            for(column = column; column < width; column++) {
                sum_0 = 0.0;

                for(i = from; i <= to; i++) {
                    sum_0 += original_image[i][row*width+column]*original_image[i][row*width+column];
                }

                new_image[channel][row*width+column] = original_image[channel][row*width+column] / powf((1+(alpha/n)*sum_0),beta);
            }
        }
    }
}


#endif


#endif // ACTIVATION_FUNCTION_H
