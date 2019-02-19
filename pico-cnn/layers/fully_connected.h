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

#endif

#endif // FULLY_CONNECTED_H
