#include "concatenate.h"

void concatenate_1D(fp_t** input_channels, const uint16_t width,
                    const uint16_t num_inputs, fp_t* output_channel){

    uint16_t input_channel;

    for(input_channel = 0; input_channel < num_inputs; input_channel++){
        memcpy(&output_channel[input_channel*width], &input_channels[input_channel], width * sizeof(fp_t));
    }
}

void concatenate_2D(fp_t** input_channels, const uint16_t height, const uint16_t width,
                    const uint16_t dimension, const uint16_t num_inputs, fp_t* output_channel) {

    uint16_t input_channel;
    uint16_t input_channel_size = width * height;

    // (height dimension)
    // probably the same as flatten
    if(dimension == 0) {

        for(input_channel = 0; input_channel < num_inputs; input_channel++){
            memcpy(&output_channel[input_channel* input_channel_size],
                   input_channels[input_channel],
                   input_channel_size * sizeof(fp_t));
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

void concatenate_naive(fp_t*** inputs, const uint16_t** input_shape, const uint16_t dimension,
                       const uint16_t num_inputs, fp_t** output_channels) {

    uint16_t input_id;
    uint16_t input_channel;

    uint16_t output_channel_counter;

    // concatenate along channels
    if(dimension == 0) {
        output_channel_counter = 0;

        for(input_id = 0; input_id < num_inputs; input_id++){

            uint16_t num_input_channels = input_shape[input_id][0];
            uint16_t input_channel_size = input_shape[input_id][1] * input_shape[input_id][2];

            for(input_channel = 0; input_channel < num_input_channels; input_channel++) {
                memcpy(output_channels[output_channel_counter + input_channel],
                       inputs[input_id][input_channel],
                       input_channel_size * sizeof(fp_t));
            }
            output_channel_counter += num_input_channels;
        }
    }

    else if(dimension == 1) {

        uint16_t output_position_counter = 0;
        uint16_t num_channels = input_shape[0][0];
        uint16_t input_channel_size;

        for(input_channel = 0; input_channel < num_channels; input_channel++) {
            for(input_id = 0; input_id < num_inputs; input_id++) {

                input_channel_size = input_shape[input_id][1] * input_shape[input_id][2];

                memcpy(&output_channels[input_channel][output_position_counter],
                       inputs[input_id][input_channel],
                       input_channel_size * sizeof(fp_t));

                output_position_counter += input_channel_size;
            }
            output_position_counter = 0;
        }
    }

    else if(dimension == 2) {

        uint16_t channel_row;
        uint16_t num_channels = input_shape[0][0];
        uint16_t num_rows = input_shape[0][1];
        uint16_t channel_width;

        uint16_t output_position_counter = 0;

        for(input_channel = 0; input_channel < num_channels; input_channel++) {
            for(channel_row = 0; channel_row < num_rows; channel_row++) {
                for(input_id = 0; input_id < num_inputs; input_id++) {

                    channel_width = input_shape[input_id][2];

                    memcpy(&output_channels[input_channel][output_position_counter],
                           &inputs[input_id][input_channel][channel_row*channel_width],
                           channel_width * sizeof(fp_t));

                    output_position_counter += channel_width;
                }
            }
            output_position_counter = 0;
        }

    } else {
        printf("ERROR: Concatenation (3-dimensional) operation not supported for dimension: %d\n", dimension);
        exit(1);
    }

}
