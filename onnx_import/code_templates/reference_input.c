#define EPSILON 0.001

#include "network.h"
#include "network_initialization.h"
#include "network_cleanup.h"

#include <stdio.h>
#include <stdlib.h>

#include "pico-cnn/pico-cnn.h"

void usage() {
    printf("./reference_input PATH_TO_BINARY_WEIGHTS_FILE PATH_TO_REFERENCE_INPUT PATH_TO_REFERENCE_OUTPUT\n");
}

int32_t almost_equal(float a, float b, float epsilon){
    return (fabs(a-b) <= epsilon);
}

int32_t main(int32_t argc, char** argv) {

    if(argc != 4) {
        usage();
        return 1;
    }

    char weights_path[1024];
    strcpy(weights_path, argv[1]);

    char sample_input_path[1024];
    strcpy(sample_input_path, argv[2]);

    char sample_output_path[1024];
    strcpy(sample_output_path, argv[3]);

    {% if input_shape_len == 4 or input_shape_len == 3 %}
    fp_t** input = (fp_t**) malloc({{num_input_channels}}*sizeof(fp_t*));

    for(uint32_t i = 0; i < {{num_input_channels}}; i++){
        input[i] = (fp_t*) malloc({{input_channel_height}}*{{input_channel_width}}*sizeof(fp_t));
    }
    {% elif input_shape_len == 2 %}
    fp_t* input = (fp_t*) malloc({{input_channel_height}}*{{input_channel_width}}*sizeof(fp_t));
    {% endif %}

    {% if output_shape_len == 4 or output_shape_len == 3 %}
    fp_t** output = (fp_t**) malloc({{num_output_channels}}*sizeof(fp_t*));

    for(uint32_t i = 0; i < {{num_input_channels}}; i++){
        output[i] = (fp_t*) malloc({{output_channel_height}}*{{output_channel_width}}*sizeof(fp_t));
    }
    {% elif output_shape_len == 2 %}
    fp_t* output = (fp_t*) malloc({{output_channel_height}}*{{output_channel_width}}*sizeof(fp_t));
    {% endif %}


    {% if output_shape_len == 4 or output_shape_len == 3 %}
    fp_t** ref_output = (fp_t**) malloc({{num_output_channels}}*sizeof(fp_t*));

    for(uint32_t i = 0; i < {{num_input_channels}}; i++){
        ref_output[i] = (fp_t*) malloc({{output_channel_height}}*{{output_channel_width}}*sizeof(fp_t));
    }
    {% elif output_shape_len == 2 %}
    fp_t* ref_output = (fp_t*) malloc({{output_channel_height}}*{{output_channel_width}}*sizeof(fp_t));
    {% endif %}

    {% if input_shape_len == 4 or input_shape_len == 3 %}
    if(read_binary_reference_input_data(sample_input_path, &input) != 0)
        return -1;
    {% elif input_shape_len == 2 %}
    fp_t** input_helper = &input;
    if(read_binary_reference_input_data(sample_input_path, &input_helper) != 0)
        return -1;
    {% endif %}
    if(read_binary_reference_output_data(sample_output_path, &ref_output) != 0)
        return -1;

    initialize_network();

    INFO_MSG("Reading weights from '%s'\n", weights_path);

    if(read_binary_weights(weights_path, &kernels, &biases) != 0){
        ERROR_MSG("Could not read weights from '%s'\n", weights_path);
        return 1;
    }

    INFO_MSG("Starting CNN...\n");

    network(input, output);

    INFO_MSG("After CNN\n");

    cleanup_network();

    int32_t all_equal = 1;

    {% if output_shape_len == 4 or output_shape_len == 3 %}
    for(uint32_t channel = 0; channel < {{num_output_channels}}; channel++) {
        for(uint32_t i = 0; i < {{output_channel_height}}*{{output_channel_width}}; i++) {

            DEBUG_MSG("Channel: %d\tPosition: %d\toutput: %f\tref_output: %f\n", channel, i, output[channel][i], ref_output[channel][i]);

            if(!almost_equal(output[channel][i], ref_output[channel][i], EPSILON)) {
                all_equal = 0;
                ERROR_MSG("Not equal at in channel: %d at position: %d, output: %f, ref_output: %f\n", channel, i, output[channel][i], ref_output[channel][i]);
            }
        }
    }
    {% elif output_shape_len == 2 %}
    for(uint32_t i = 0; i < {{output_channel_height}}*{{output_channel_width}}; i++) {

        DEBUG_MSG("Position: %d\toutput: %f\tref_output: %f\n", i, output[i], ref_output[i]);

        if(!almost_equal(output[i], ref_output[i], EPSILON)) {
            all_equal = 0;
            ERROR_MSG("Not equal at position: %d, output: %f, ref_output: %f\n", i, output[i], ref_output[i]);
        }
    }
    {% endif %}

    if(all_equal) {
        INFO_MSG("Output is almost equal (epsilon=%f) to reference output!\n", EPSILON);
    } else {
        ERROR_MSG("WARNING: Output is not almost equal (epsilon=%f) to reference output!\n", EPSILON);
    }

    free(output);
    free(ref_output);

    {% if input_shape_len == 4 or input_shape_len == 3 %}
    for(uint32_t i = 0; i < {{num_input_channels}}; i++) {
        free(input[i]);
    }
    {% endif %}

    free(input);

    return 0;

}