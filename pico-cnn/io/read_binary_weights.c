
#include "read_binary_weights.h"


int read_binary_weights(const char* path_to_weights_file, fp_t**** kernels, fp_t*** biases) {

    FILE *binary_file;
    binary_file = fopen(path_to_weights_file, "r");

    if(binary_file != 0) {

        // Read magic number
        char magic_number[4];
        if(fread((void*)&magic_number, 1, 3, binary_file) != 3) {
            printf("Error reading magic number\n");
            fclose(binary_file);
            return 1;
        }
        magic_number[3] = '\0';

        if(strcmp(magic_number,"FD\n") != 0) {
            printf("Wrong magic number: %s\n", magic_number);
            fclose(binary_file);
            return 1;
        }
        #ifdef DEBUG
        printf("%s", magic_number);
        #endif

        // Read name
        char buffer[100];
        char c = '\0';
        uint32_t i = 0;
        for(; c != '\n'; i++){
            if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                printf("Error while reading name\n");
                fclose(binary_file);
                return 1;
            }
            buffer[i] = c;
        }
        buffer[i-1] = '\0'; //terminate string correctly

        #ifdef DEBUG
        printf("%s\n", buffer);
        #endif

        // Read number of layers
        uint32_t num_layers;
        if(fread((void*)&num_layers, sizeof(num_layers), 1, binary_file) != 1) {
            printf("ERROR reading number of layers\n");
            fclose(binary_file);
            return 1;
        } else {
            #ifdef DEBUG
            printf("Number of layers: %d\n", num_layers);
            #endif
        }

        for(uint32_t layer = 0; layer < num_layers; layer++) {

            c = '\0';
            i = 0;
            for(; c != '\n'; i++) {
                if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                    printf("Error while reading layer name\n");
                    fclose(binary_file);
                    return 1;
                }
                buffer[i] = c;
            }
            buffer[i-1] = '\0'; //terminate string correctly

            #ifdef DEBUG
            printf("Layer %d: %s\n", layer, buffer);
            #endif

            uint32_t kernel_height = 0;
            uint32_t kernel_width = 0;
            uint32_t num_kernels = 0;

            if(fread((void*)&kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                printf("Error while reading kernel height\n");
                fclose(binary_file);
                return 1;
            }
            if(fread((void*)&kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                printf("Error while reading kernel width\n");
                fclose(binary_file);
                return 1;
            }
            if(fread((void*)&num_kernels, sizeof(num_kernels), 1, binary_file) != 1) {
                printf("Error while reading number of kernels\n");
                fclose(binary_file);
                return 1;
            }

            #ifdef DEBUG
            printf("Height: %d, width: %d, num: %d\n", kernel_height, kernel_width, num_kernels);
            #endif

            uint32_t kernel;
            float *values = (float*) malloc(kernel_height*kernel_width*sizeof(float));

            int numbers_read = -1;

            for(kernel = 0; kernel < num_kernels; kernel++) {
                numbers_read = fread((void*)values, sizeof(float), kernel_height*kernel_width, binary_file);
                memcpy((*kernels)[layer][kernel], values, kernel_height*kernel_width*sizeof(float));
            }

            free(values);

            uint32_t num_biases = 0;
            if(fread((void*)&num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                printf("Error while reading number of biases\n");
                fclose(binary_file);
                return 1;
            }

            #ifdef DEBUG
            printf("Number of biases: %d\n", num_biases);
            #endif

            uint32_t bias;
            float *bias_values = (float*) malloc(num_biases*sizeof(float));

            fread((void*)bias_values, sizeof(float), num_biases, binary_file);
            memcpy((*biases)[layer], bias_values, num_biases*sizeof(float));

            free(bias_values);

        }
    }
    fclose(binary_file);
    return 0;

}
