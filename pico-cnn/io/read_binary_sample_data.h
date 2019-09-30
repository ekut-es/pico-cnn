//
// Created by junga on 27.08.19.
//

#ifndef PICO_CNN_READ_BINARY_SAMPLE_DATA_H
#define PICO_CNN_READ_BINARY_SAMPLE_DATA_H

#include "../parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int read_binary_sample_input_data(const char* path_to_sample_data, fp_t*** input) {

    FILE *binary_file;
    binary_file = fopen(path_to_sample_data, "r");

    if(binary_file != 0) {

        // Read magic number
        char magic_number[5];
        if(fread((void*)&magic_number, 1, 4, binary_file) != 4) {
            printf("Error reading magic number of input file.\n");
            fclose(binary_file);
            return 1;
        }
        magic_number[4] = '\0';

        if(strcmp(magic_number,"FCI\n") != 0) {
            printf("Wrong magic number: %s\n", magic_number);
            fclose(binary_file);
            return 1;
        }
        #ifdef DEBUG
        printf("%s", magic_number);
        #endif

        // Read number of channels
        uint32_t num_channels;
        if(fread((void*)&num_channels, sizeof(num_channels), 1, binary_file) != 1) {
            printf("ERROR reading number of channels\n");
            fclose(binary_file);
            return 1;
        } else {
            #ifdef DEBUG
            printf("Number of channels: %d\n", num_channels);
            #endif
        }

        // Read channel width
        uint32_t width;
        if(fread((void*)&width, sizeof(width), 1, binary_file) != 1) {
            printf("ERROR reading width\n");
            fclose(binary_file);
            return 1;
        } else {
            #ifdef DEBUG
            printf("Width: %d\n", width);
            #endif
        }

        // Read channel height
        uint32_t height;
        if(fread((void*)&height, sizeof(height), 1, binary_file) != 1) {
            printf("ERROR reading height\n");
            fclose(binary_file);
            return 1;
        } else {
            #ifdef DEBUG
            printf("Height: %d\n", height);
            #endif
        }

        for(uint32_t channel = 0; channel < num_channels; channel++) {

            float *values = (float*) malloc(width*height*sizeof(float));

            int numbers_read = -1;

            numbers_read = fread((void*)values, sizeof(float), width*height, binary_file);
            memcpy((*input)[channel], values, width*height*sizeof(float));

            free(values);
        }
    }
    fclose(binary_file);

    return 0;
}

int read_binary_sample_output_data(const char* path_to_sample_data, fp_t** output) {

    FILE *binary_file;
    binary_file = fopen(path_to_sample_data, "r");

    if(binary_file != 0) {

        // Read magic number
        char magic_number[5];
        if(fread((void*)&magic_number, 1, 4, binary_file) != 4) {
            printf("Error reading magic number of output file.\n");
            fclose(binary_file);
            return 1;
        }
        magic_number[4] = '\0';

        if(strcmp(magic_number,"FCO\n") != 0) {
            printf("Wrong magic number: %s\n", magic_number);
            fclose(binary_file);
            return 1;
        }
        #ifdef DEBUG
        printf("%s", magic_number);
        #endif

        // Read number of outputs
        uint32_t num_outputs;
        if(fread((void*)&num_outputs, sizeof(num_outputs), 1, binary_file) != 1) {
            printf("ERROR reading number of outputs\n");
            fclose(binary_file);
            return 1;
        } else {
            #ifdef DEBUG
            printf("Number of outputs: %d\n", num_outputs);
            #endif
        }

        float *values = (float*) malloc(num_outputs*sizeof(float));

        int numbers_read = -1;

        numbers_read = fread((void*)values, sizeof(float), num_outputs, binary_file);
        memcpy((*output), values, num_outputs*sizeof(float));

        free(values);

    }
    fclose(binary_file);

    return 0;
}

#endif //PICO_CNN_READ_BINARY_SAMPLE_DATA_H
