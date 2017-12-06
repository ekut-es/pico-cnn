/** 
 * @brief contains local response normalization
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef LOCAL_RESPONSE_NORMALIZATION_H
#define LOCAL_RESPONSE_NORMALIZATION_H

#include "../parameters.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>

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
void local_response_normalization_naive(const fp_t** original_image, const uint16_t height, const uint16_t width, const uint16_t depth, fp_t** new_image, const fp_t alpha, const fp_t beta, const uint16_t n) {
    
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

#endif // LOCAL_RESPONSE_NORMALIZATION_H
