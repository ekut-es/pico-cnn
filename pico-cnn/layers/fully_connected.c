#include "fully_connected.h"

void fully_connected_naive(const fp_t* input_channel, const uint16_t input_width, fp_t* output_channel, const uint16_t output_width, const fp_t* kernel, const fp_t* bias) {

    int i, j;
    for(i = 0; i < output_width; i++) {

        fp_t pixel = 0.0;

        for(j = 0; j < input_width; j++) {
            // takes each output_width'nd element
            pixel += input_channel[j] * kernel[j*output_width+i];
        }

        if(bias)
            pixel += bias[i];

        output_channel[i] = pixel;
    }
}
