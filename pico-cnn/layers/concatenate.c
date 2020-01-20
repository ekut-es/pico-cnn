#include "concatenate.h"

/*  concatenates 1d channels into a single channel
 *  output channel needs to be of size width  * num_inputs i * sizeof(fp_t)
 */
void concatenate_1D(fp_t** input_channels, uint16_t width, uint16_t num_inputs, fp_t* output_channel){

    uint16_t input_channel;

    for(input_channel = 0; input_channel < num_inputs; input_channel++){
        memcpy(&output_channel[input_channel*width ], &input_channels[input_channel], width  * sizeof(fp_t));
    }
}


void concatenate_2D(fp_t** input_channels, uint16_t width, uint16_t height,
               uint16_t dimension, uint16_t num_inputs, fp_t* output_channel) {

    uint16_t input_channel;

    // (height dimension)
    // probably the same as flatten
    if(dimension == 0) {

        for(input_channel = 0; input_channel < num_inputs; input_channel++){
            memcpy(&output_channel[input_channel* width *height],
                   input_channels[input_channel],
                   width  * height * sizeof(fp_t));
        }

    }

    // (width dimension)
    else if(dimension == 1) {

        uint16_t row;
        for(row = 0; row < height; row++) {
            for(input_channel = 0; input_channel < num_inputs; input_channel++) {
                memcpy(&output_channel[input_channel*width + num_inputs*width*row],
                       &input_channels[input_channel][row*width],
                       width *sizeof(fp_t));
            }
        }
    }
}
