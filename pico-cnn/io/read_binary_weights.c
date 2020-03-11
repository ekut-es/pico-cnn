#include "read_binary_weights.h"

int32_t read_binary_weights(const char* path_to_weights_file, fp_t**** kernels, fp_t*** biases) {

    FILE *binary_file;
    binary_file = fopen(path_to_weights_file, "r");

    if(binary_file != 0) {

        // Read magic number
        char magic_number[4];
        if(fread((void*)&magic_number, 1, 3, binary_file) != 3) {
            ERROR_MSG("ERROR reading magic number\n");
            fclose(binary_file);
            return 1;
        }
        magic_number[3] = '\0';

        if(strcmp(magic_number,"FD\n") != 0) {
            ERROR_MSG("ERROR: Wrong magic number: %s\n", magic_number);
            fclose(binary_file);
            return 1;
        }
        DEBUG_MSG("%s", magic_number);

        // Read name
        char buffer[100];
        char c = '\0';
        uint32_t i = 0;
        for(; c != '\n'; i++){
            if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                ERROR_MSG("ERROR while reading name\n");
                fclose(binary_file);
                return 1;
            }
            buffer[i] = c;
        }
        buffer[i-1] = '\0'; //terminate string correctly

        DEBUG_MSG("%s\n", buffer);

        // Read number of layers
        uint32_t num_layers;
        if(fread((void*)&num_layers, sizeof(num_layers), 1, binary_file) != 1) {
            ERROR_MSG("ERROR reading number of layers\n");
            fclose(binary_file);
            return 1;
        } else {
            DEBUG_MSG("Number of layers: %d\n", num_layers);
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
                    ERROR_MSG("ERROR while reading layer name\n");
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
                    ERROR_MSG("ERROR while reading layer type\n");
                    fclose(binary_file);
                    return 1;
                }
                buffer_layer_type[i] = c;
            }
            buffer_layer_type[i-1] = '\0'; //terminate string correctly

            DEBUG_MSG("Layer %d: %s of type: %s\n", layer, buffer, buffer_layer_type);

            if(strcmp(buffer_layer_type, "Conv") == 0) {

                uint32_t kernel_height = 0;
                uint32_t kernel_width = 0;
                uint32_t num_kernels = 0;

                if (fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading kernel height\n");
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading kernel width\n");
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &num_kernels, sizeof(num_kernels), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of kernels\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Height: %d, width: %d, num: %d, kernel_idx: %d\n", kernel_height, kernel_width, num_kernels, kernel_idx);

                if(kernel_height != 0 && kernel_width != 0 && num_kernels != 0) {
                    uint32_t kernel;
                    float *values = (float *) malloc(kernel_height * kernel_width * sizeof(float));

                    for (kernel = 0; kernel < num_kernels; kernel++) {
                        if(fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file) != (kernel_height*kernel_width)) {
                            ERROR_MSG("ERROR while reading kernel values.\n");
                            free(values);
                            fclose(binary_file);
                            return 1;
                        }
                        memcpy((*kernels)[kernel_idx][kernel], values, kernel_height * kernel_width * sizeof(float));
                    }

                    kernel_idx++;

                    free(values);
                }

                uint32_t num_biases = 0;
                if (fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of biases\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Number of biases: %d\n", num_biases);

                if (num_biases) {
                    float *bias_values = (float *) malloc(num_biases * sizeof(float));

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        ERROR_MSG("ERROR while reading bias values.\n");
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    memcpy((*biases)[bias_idx], bias_values, num_biases * sizeof(float));

                    bias_idx++;

                    free(bias_values);
                }

            } else if (strcmp(buffer_layer_type, "BatchNormalization") == 0) {
                // read gamma values
                uint32_t num_gamma = 0;
                if(fread((void *) &num_gamma, sizeof(num_gamma), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of gamma values\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Number of gamma values: %d\n", num_gamma);

                if(num_gamma) {
                    float *gamma_values = (float *) malloc(num_gamma * sizeof(float));

                    if(fread((void *) gamma_values, sizeof(float), num_gamma, binary_file) != num_gamma) {
                        ERROR_MSG("ERROR while reading gamma values.\n");
                        free(gamma_values);
                        fclose(binary_file);
                        return 1;
                    }
                    memcpy((*biases)[bias_idx], gamma_values, num_gamma * sizeof(float));

                    bias_idx++;

                    free(gamma_values);
                }

                // read beta values
                uint32_t num_beta = 0;
                if(fread((void *) &num_beta, sizeof(num_beta), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of beta values\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Number of beta values: %d\n", num_beta);

                if(num_beta) {
                    float *beta_values = (float *) malloc(num_beta * sizeof(float));

                    if(fread((void *) beta_values, sizeof(float), num_beta, binary_file) != num_beta) {
                        ERROR_MSG("ERROR while reading beta values.\n");
                        free(beta_values);
                        fclose(binary_file);
                        return 1;
                    }
                    memcpy((*biases)[bias_idx], beta_values, num_beta * sizeof(float));

                    bias_idx++;

                    free(beta_values);
                }

                // read mean values
                uint32_t num_mean = 0;
                if(fread((void *) &num_mean, sizeof(num_mean), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of beta values\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Number of mean values: %d\n", num_mean);

                if(num_mean) {
                    float *mean_values = (float *) malloc(num_mean * sizeof(float));

                    if(fread((void *) mean_values, sizeof(float), num_mean, binary_file) != num_mean) {
                        ERROR_MSG("ERROR while reading mean values.\n");
                        free(mean_values);
                        fclose(binary_file);
                        return 1;
                    }
                    memcpy((*biases)[bias_idx], mean_values, num_mean * sizeof(float));

                    bias_idx++;

                    free(mean_values);
                }

                // read variance values
                uint32_t num_variance = 0;
                if(fread((void *) &num_variance, sizeof(num_variance), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of variance values\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Number of variance values: %d\n", num_variance);

                if(num_variance) {
                    float *variance_values = (float *) malloc(num_variance * sizeof(float));

                    if(fread((void *) variance_values, sizeof(float), num_variance, binary_file) != num_variance) {
                        ERROR_MSG("ERROR while reading variance values.\n");
                        free(variance_values);
                        fclose(binary_file);
                        return 1;
                    }
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

                if(fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading kernel height\n");
                    fclose(binary_file);
                    return 1;
                }
                if(fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading kernel width\n");
                    fclose(binary_file);
                    return 1;
                }
                if(fread((void *) &num_kernels, sizeof(num_kernels), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of kernels\n");
                    fclose(binary_file);
                    return 1;
                }

                DEBUG_MSG("Height: %d, width: %d, num: %d, kernel_idx: %d\n", kernel_height, kernel_width, num_kernels, kernel_idx);

                uint32_t kernel;
                float *values = (float *) malloc(kernel_height * kernel_width * sizeof(float));

                for(kernel = 0; kernel < num_kernels; kernel++) {
                    if(fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file) != (kernel_height*kernel_width)) {
                        ERROR_MSG("ERROR while reading kernel values.\n");
                        free(values);
                        fclose(binary_file);
                        return 1;
                    }
                    memcpy((*kernels)[kernel_idx][kernel], values, kernel_height * kernel_width * sizeof(float));
                }

                kernel_idx++;

                free(values);

                uint32_t num_biases = 0;
                if(fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of biases\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Number of biases: %d\n", num_biases);

                if(num_biases) {
                    float *bias_values = (float *) malloc(num_biases * sizeof(float));

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        ERROR_MSG("ERROR while reading bias values.\n");
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    memcpy((*biases)[bias_idx], bias_values, num_biases * sizeof(float));

                    bias_idx++;

                    free(bias_values);
                }
            } else if (strcmp(buffer_layer_type, "Add") == 0) {
                uint32_t num_biases = 0;
                if(fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    ERROR_MSG("ERROR while reading number of biases\n");
                    fclose(binary_file);
                    return 1;
                }
                DEBUG_MSG("Number of biases: %d\n", num_biases);

                if (num_biases) {
                    float *bias_values = (float *) malloc(num_biases * sizeof(float));

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        ERROR_MSG("ERROR while reading bias values.\n");
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    memcpy((*biases)[bias_idx], bias_values, num_biases * sizeof(float));

                    bias_idx++;

                    free(bias_values);
                }
            } else {
                ERROR_MSG("ERROR: Unknown layer type \"%s\" in weights file. Layer number: %d\n", buffer_layer_type, layer);
                fclose(binary_file);
                return 1;
            }

        }
        DEBUG_MSG("Layer idx: %d, kernel idx: %d, bias idx: %d\n", layer, kernel_idx, bias_idx);

        // Read end marker
        char end_marker[5];
        if(fread((void*)&end_marker, 1, 4, binary_file) != 4) {
            ERROR_MSG("ERROR reading end marker\n");
            fclose(binary_file);
            return 1;
        }
        end_marker[4] = '\0';

        if(strcmp(end_marker,"end\n") != 0) {
            ERROR_MSG("ERROR: Wrong end marker read: %s\n", end_marker);
            fclose(binary_file);
            return 1;
        }
        DEBUG_MSG("%s", end_marker);

    }
    fclose(binary_file);
    return 0;

}
