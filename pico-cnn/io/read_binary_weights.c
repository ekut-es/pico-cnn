
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

        // With the support for BatchNormalization we need separate counters
        // for kernels and biases as the BatchNormalization layer only has
        // four bias like arrays of values.
        uint32_t layer, kernel_idx, bias_idx;
        kernel_idx = 0;
        bias_idx = 0;

        for(layer = 0; layer < num_layers; layer++) {

            // Read layer name
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

            // Read layer type
            char buffer_layer_type[100];
            c = '\0';
            i = 0;
            for(; c != '\n'; i++) {
                if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                    printf("Error while reading layer type\n");
                    fclose(binary_file);
                    return 1;
                }
                buffer_layer_type[i] = c;
            }
            buffer_layer_type[i-1] = '\0'; //terminate string correctly

            #ifdef DEBUG
            printf("Layer %d: %s of type: %s\n", layer, buffer, buffer_layer_type);
            #endif

            if(strcmp(buffer_layer_type, "Conv") == 0) {

                uint32_t kernel_height = 0;
                uint32_t kernel_width = 0;
                uint32_t num_kernels = 0;

                if (fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    printf("Error while reading kernel height\n");
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    printf("Error while reading kernel width\n");
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &num_kernels, sizeof(num_kernels), 1, binary_file) != 1) {
                    printf("Error while reading number of kernels\n");
                    fclose(binary_file);
                    return 1;
                }

                #ifdef DEBUG
                printf("Height: %d, width: %d, num: %d\n", kernel_height, kernel_width, num_kernels);
                #endif

                uint32_t kernel;
                float *values = (float *) malloc(kernel_height * kernel_width * sizeof(float));

                int numbers_read = -1;

                for (kernel = 0; kernel < num_kernels; kernel++) {
                    numbers_read = fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file);
                    memcpy((*kernels)[kernel_idx][kernel], values, kernel_height * kernel_width * sizeof(float));
                }

                kernel_idx++;

                free(values);

                uint32_t num_biases = 0;
                if (fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    printf("Error while reading number of biases\n");
                    fclose(binary_file);
                    return 1;
                }
                #ifdef DEBUG
                printf("Number of biases: %d\n", num_biases);
                #endif

                if (num_biases) {
                    float *bias_values = (float *) malloc(num_biases * sizeof(float));

                    fread((void *) bias_values, sizeof(float), num_biases, binary_file);
                    memcpy((*biases)[bias_idx], bias_values, num_biases * sizeof(float));

                    bias_idx++;

                    free(bias_values);
                }

            } else if (strcmp(buffer_layer_type, "BatchNormalization") == 0) {
                // read gamma values
                uint32_t num_gamma = 0;
                if (fread((void *) &num_gamma, sizeof(num_gamma), 1, binary_file) != 1) {
                    printf("Error while reading number of gamma values\n");
                    fclose(binary_file);
                    return 1;
                }
#ifdef DEBUG
                printf("Number of gamma values: %d\n", num_gamma);
#endif
                if(num_gamma) {
                    float *gamma_values = (float *) malloc(num_gamma * sizeof(float));

                    fread((void *) gamma_values, sizeof(float), num_gamma, binary_file);
                    memcpy((*biases)[bias_idx], gamma_values, num_gamma * sizeof(float));

                    bias_idx++;

                    free(gamma_values);
                }

                // read beta values
                uint32_t num_beta = 0;
                if (fread((void *) &num_beta, sizeof(num_beta), 1, binary_file) != 1) {
                    printf("Error while reading number of beta values\n");
                    fclose(binary_file);
                    return 1;
                }
#ifdef DEBUG
                printf("Number of beta values: %d\n", num_beta);
#endif
                if(num_beta) {
                    float *beta_values = (float *) malloc(num_beta * sizeof(float));

                    fread((void *) beta_values, sizeof(float), num_beta, binary_file);
                    memcpy((*biases)[bias_idx], beta_values, num_beta * sizeof(float));

                    bias_idx++;

                    free(beta_values);
                }

                // read mean values
                uint32_t num_mean = 0;
                if (fread((void *) &num_mean, sizeof(num_mean), 1, binary_file) != 1) {
                    printf("Error while reading number of beta values\n");
                    fclose(binary_file);
                    return 1;
                }
#ifdef DEBUG
                printf("Number of mean values: %d\n", num_mean);
#endif
                if(num_mean) {
                    float *mean_values = (float *) malloc(num_mean * sizeof(float));

                    fread((void *) mean_values, sizeof(float), num_mean, binary_file);
                    memcpy((*biases)[bias_idx], mean_values, num_mean * sizeof(float));

                    bias_idx++;

                    free(mean_values);
                }

                // read variance values
                uint32_t num_variance = 0;
                if (fread((void *) &num_variance, sizeof(num_variance), 1, binary_file) != 1) {
                    printf("Error while reading number of variance values\n");
                    fclose(binary_file);
                    return 1;
                }
#ifdef DEBUG
                printf("Number of variance values: %d\n", num_variance);
#endif
                if(num_variance) {
                    float *variance_values = (float *) malloc(num_variance * sizeof(float));

                    fread((void *) variance_values, sizeof(float), num_variance, binary_file);
                    memcpy((*biases)[bias_idx], variance_values, num_variance * sizeof(float));

                    bias_idx++;

                    free(variance_values);
                }

            } else if (strcmp(buffer_layer_type, "Gemm") == 0 ||
                       strcmp(buffer_layer_type, "MatMul") == 0 ||
                       strcmp(buffer_layer_type, "Transpose") == 0) {

                uint32_t kernel_height = 0;
                uint32_t kernel_width = 0;
                uint32_t num_kernels = 0;

                if (fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    printf("Error while reading kernel height\n");
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    printf("Error while reading kernel width\n");
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &num_kernels, sizeof(num_kernels), 1, binary_file) != 1) {
                    printf("Error while reading number of kernels\n");
                    fclose(binary_file);
                    return 1;
                }

#ifdef DEBUG
                printf("Height: %d, width: %d, num: %d\n", kernel_height, kernel_width, num_kernels);
#endif

                uint32_t kernel;
                float *values = (float *) malloc(kernel_height * kernel_width * sizeof(float));

                int numbers_read = -1;

                for (kernel = 0; kernel < num_kernels; kernel++) {
                    numbers_read = fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file);
                    memcpy((*kernels)[kernel_idx][kernel], values, kernel_height * kernel_width * sizeof(float));
                }

                kernel_idx++;

                free(values);

                uint32_t num_biases = 0;
                if (fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    printf("Error while reading number of biases\n");
                    fclose(binary_file);
                    return 1;
                }
#ifdef DEBUG
                printf("Number of biases: %d\n", num_biases);
#endif

                if (num_biases) {
                    float *bias_values = (float *) malloc(num_biases * sizeof(float));

                    fread((void *) bias_values, sizeof(float), num_biases, binary_file);
                    memcpy((*biases)[bias_idx], bias_values, num_biases * sizeof(float));

                    bias_idx++;

                    free(bias_values);
                }
            } else if (strcmp(buffer_layer_type, "Add") == 0) {
                uint32_t num_biases = 0;
                if (fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    printf("Error while reading number of biases\n");
                    fclose(binary_file);
                    return 1;
                }
#ifdef DEBUG
                printf("Number of biases: %d\n", num_biases);
#endif

                if (num_biases) {
                    float *bias_values = (float *) malloc(num_biases * sizeof(float));

                    fread((void *) bias_values, sizeof(float), num_biases, binary_file);
                    memcpy((*biases)[bias_idx], bias_values, num_biases * sizeof(float));

                    bias_idx++;

                    free(bias_values);
                }
            } else {
                printf("ERROR: Unknown layer type \"%s\" in weights file. Layer number: %d\n", buffer_layer_type, layer);
                fclose(binary_file);
                return 1;
            }

        }
        #ifdef DEBUG
        printf("Layer idx: %d, kernel idx: %d, bias idx: %d\n", layer, kernel_idx, bias_idx);
        #endif

        // Read end marker
        char end_marker[5];
        if(fread((void*)&end_marker, 1, 4, binary_file) != 4) {
            printf("Error reading end marker\n");
            fclose(binary_file);
            return 1;
        }
        end_marker[4] = '\0';

        if(strcmp(end_marker,"end\n") != 0) {
            printf("Wrong magic number: %s\n", end_marker);
            fclose(binary_file);
            return 1;
        }
#ifdef DEBUG
        printf("%s", magic_number);
#endif

    }
    fclose(binary_file);
    return 0;

}
