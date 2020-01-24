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
                        width * sizeof(fp_t));
            }
        }
    }
}

void concatenate_3D(fp_t*** inputs, uint16_t channel_width, uint16_t channel_height,
                    uint16_t dimension, uint16_t num_inputs, uint16_t num_input_channels,
                    fp_t** output_channels){
    uint16_t input_id;
    uint16_t input_channel;

    // concatenate along inputs
    if(dimension == 0) {
        for(input_id = 0; input_id < num_inputs; input_id++){
           for(input_channel = 0; input_channel < num_input_channels; input_channel++) {
              memcpy(output_channels[input_id * num_input_channels + input_channel],
                     inputs[input_id][input_channel],
                     channel_width * channel_height * sizeof(fp_t));
           }
        }


    }

    // concatenate along channels
    else if(dimension == 1) {
        for(input_channel = 0; input_channel < num_input_channels; input_channel++) {
            for(input_id = 0; input_id < num_inputs; input_id++) {
                memcpy(&output_channels[input_channel][input_id*channel_width*channel_height],
                        inputs[input_id][input_channel],
                        channel_width * channel_height * sizeof(fp_t));
            }
        }
    }


    // concatenate along channel rows
    else if(dimension == 2) {
        uint16_t channel_row;
        for(input_channel = 0; input_channel < num_input_channels; input_channel++) {
            for(channel_row = 0; channel_row < channel_height; channel_row++) {
                for(input_id = 0; input_id < num_inputs; input_id++) {
                    memcpy(&output_channels[input_channel][input_id*channel_width + num_inputs*channel_row*channel_width],
                           &inputs[input_id][input_channel][channel_row*channel_width],
                            channel_width * sizeof(fp_t));
                }
            }

        }
    }


}
