#include "parameters.h"
#include <stdint.h>
#include <math.h>

/**
 * @brief applies tanh(x) to all pixel of original_image and stores it in
 * new_image
 *
 * @param original_image (height x width)
 * @param height
 * @param width
 * @param new_image (height x width)
 */
void tanh_naive(const float_t* original_image, const uint16_t height, const uint16_t width, float_t* new_image) {

    uint16_t i;

    for(i = 0; i < height*width; i++) {
        new_image[i] = tanhf(original_image[i]);
    }
}
