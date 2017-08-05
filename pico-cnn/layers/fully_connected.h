/** 
 * @brief contains the fully connected layer
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#include "../parameters.h"
#include <stdint.h>

/**
 * @brief implementation of fully connected layer
 *
 * @param original_image (1 x original_width)
 * @param original_width
 * @param new_image (1 x new_width
 * @param new_width
 * @param kernel
 * @param bias
 */
void fully_connected_naive(const float_t* original_image, const uint16_t original_width, float_t* new_image, const uint16_t new_width, const float_t* kernel, const float_t bias) {

    int i, j;
    for(i = 0; i < new_width; i++) {

        float_t pixel = 0.0;

        for(j = 0; j < original_width; j++) {
            //pixel += original_image[j]*kernel[i*original_width+j];
            pixel += original_image[j]*kernel[j];
        }

        pixel += bias;
        new_image[i] = pixel;
    }
}
