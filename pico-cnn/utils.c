#include "utils.h"

void extend_2d_input_with_padding(const fp_t* input_channel, const uint16_t height, const uint16_t width,
                               fp_t** extended_input, const int* padding, fp_t initializer) {

    uint16_t height_padded = height + padding[0] + padding[2];
    uint16_t width_padded = width + padding[1] + padding[3];

    *extended_input = (fp_t*) calloc(height_padded * width_padded, sizeof(fp_t));

    if(initializer != 0.0) {
        for(int r = 0; r < height_padded; r++){
            for(int c = 0; c < width_padded; c++){
                (*extended_input)[r*width_padded+c] = initializer;
            }
        }
    }

    for (int16_t channel_row = 0; channel_row < height; channel_row++) {
        memcpy((*extended_input) + (channel_row + padding[0]) * width_padded + padding[1],
               input_channel + channel_row * width, width * sizeof(fp_t));
    }
}

void extend_1d_input_with_padding(const fp_t* input_channel, const uint16_t width,
                                  fp_t** extended_input, const int* padding, fp_t initializer) {

    uint16_t width_padded = width + padding[0] + padding[1];

    *extended_input = (fp_t*) calloc(width_padded, sizeof(fp_t));

    if(initializer != 0.0) {
        for(int i = 0; i < width_padded; i++){
            (*extended_input)[i] = initializer;
        }
    }

    memcpy((*extended_input)+padding[0], input_channel, width*sizeof(fp_t));
}
