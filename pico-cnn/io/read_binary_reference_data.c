#include "read_binary_reference_data.h"

int32_t read_binary_reference_input_data(const char* path_to_sample_data, fp_t*** input) {

    FILE *binary_file;
    binary_file = fopen(path_to_sample_data, "r");

    if(binary_file != 0) {

        // Read magic number
        char magic_number[5];
        if(fread((void*)&magic_number, 1, 4, binary_file) != 4) {
            ERROR_MSG("ERROR reading magic number of input file.\n");
            fclose(binary_file);
            return 1;
        }
        magic_number[4] = '\0';

        if(strcmp(magic_number,"FCI\n") != 0) {
            ERROR_MSG("ERROR: Wrong magic number: %s\n", magic_number);
            fclose(binary_file);
            return 1;
        }
        DEBUG_MSG("%s", magic_number);

        // Read number of channels
        uint32_t num_channels;
        if(fread((void*)&num_channels, sizeof(num_channels), 1, binary_file) != 1) {
            ERROR_MSG("ERROR reading number of channels\n");
            fclose(binary_file);
            return 1;
        } else {
            DEBUG_MSG("Number of channels: %d\n", num_channels);
        }

        // Read channel height
        uint32_t height;
        if(fread((void*)&height, sizeof(height), 1, binary_file) != 1) {
            ERROR_MSG("ERROR reading height\n");
            fclose(binary_file);
            return 1;
        } else {
            DEBUG_MSG("Height: %d\n", height);
        }

        // Read channel width
        uint32_t width;
        if(fread((void*)&width, sizeof(width), 1, binary_file) != 1) {
            ERROR_MSG("ERROR reading width\n");
            fclose(binary_file);
            return 1;
        } else {
            DEBUG_MSG("Width: %d\n", width);
        }

        for(uint32_t channel = 0; channel < num_channels; channel++) {

            float *values = (float*) malloc(height*width*sizeof(float));

            int32_t numbers_read = -1;

            numbers_read = fread((void*)values, sizeof(float), height*width, binary_file);

            if(numbers_read != height*width) {
                ERROR_MSG("ERROR reading data. numbers_read = %d\n", numbers_read);
                free(values);
                return 1;
            } else {
                #ifdef DEBUG
                for(uint32_t i = 0; i < height*width; i++) {
                    DEBUG_MSG("%f\n", values[i]);
                }
                #endif
                memcpy((*input)[channel], values, height*width*sizeof(float));
            }

            free(values);
        }
    }
    fclose(binary_file);

    return 0;
}

int32_t read_binary_reference_output_data(const char* path_to_sample_data, fp_t** output) {

    FILE *binary_file;
    binary_file = fopen(path_to_sample_data, "r");

    if(binary_file != 0) {

        // Read magic number
        char magic_number[5];
        if(fread((void*)&magic_number, 1, 4, binary_file) != 4) {
            ERROR_MSG("ERROR reading magic number of output file.\n");
            fclose(binary_file);
            return 1;
        }
        magic_number[4] = '\0';

        if(strcmp(magic_number,"FCO\n") != 0) {
            ERROR_MSG("ERROR: Wrong magic number: %s\n", magic_number);
            fclose(binary_file);
            return 1;
        }
        DEBUG_MSG("%s", magic_number);

        // Read number of outputs
        uint32_t num_outputs;
        if(fread((void*)&num_outputs, sizeof(num_outputs), 1, binary_file) != 1) {
            ERROR_MSG("ERROR reading number of outputs\n");
            fclose(binary_file);
            return 1;
        } else {
            DEBUG_MSG("Number of outputs: %d\n", num_outputs);
        }

        float *values = (float*) malloc(num_outputs*sizeof(float));

        if(fread((void*)values, sizeof(float), num_outputs, binary_file) != num_outputs) {
            ERROR_MSG("ERROR reading output values.\n");
            free(values);
            fclose(binary_file);
            return 1;
        }

        memcpy((*output), values, num_outputs*sizeof(float));

        free(values);

    }
    fclose(binary_file);

    return 0;
}
