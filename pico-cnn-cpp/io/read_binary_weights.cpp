#include "read_binary_weights.h"

int32_t read_binary_weights(const char* path_to_weights_file, pico_cnn::naive::Tensor **kernels, pico_cnn::naive::Tensor **biases) {

    FILE *binary_file;
    binary_file = fopen(path_to_weights_file, "r");

    if(binary_file != 0) {

        // Read magic number
        char magic_number[4];
        if(fread((void*)&magic_number, 1, 3, binary_file) != 3) {
            PRINT_ERROR("ERROR reading magic number")
            fclose(binary_file);
            return 1;
        }
        magic_number[3] = '\0';

        if(strcmp(magic_number,"FD\n") != 0) {
            PRINT_ERROR("ERROR: Wrong magic number: " << magic_number)
            fclose(binary_file);
            return 1;
        }
        PRINT_DEBUG(magic_number)

        // Read name
        char buffer[100];
        char c = '\0';
        uint32_t i = 0;
        for(; c != '\n'; i++){
            if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                PRINT_ERROR("ERROR while reading name")
                fclose(binary_file);
                return 1;
            }
            buffer[i] = c;
        }
        buffer[i-1] = '\0'; //terminate string correctly

        PRINT_DEBUG(buffer)

        // Read number of layers
        uint32_t num_layers;
        if(fread((void*)&num_layers, sizeof(num_layers), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading number of layers")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Number of layers: " << num_layers)
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
                    PRINT_ERROR("ERROR while reading layer name")
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
                    PRINT_ERROR("ERROR while reading layer type")
                    fclose(binary_file);
                    return 1;
                }
                buffer_layer_type[i] = c;
            }
            buffer_layer_type[i-1] = '\0'; //terminate string correctly

            PRINT_DEBUG("Layer " << layer << ": " << buffer << " of type: " << buffer_layer_type)

            if(strcmp(buffer_layer_type, "Conv") == 0) {

                uint32_t num_output_channels = 0;
                uint32_t num_input_channels = 0;
                uint32_t kernel_height = 0;
                uint32_t kernel_width = 0;

                if (fread((void *) &num_output_channels, sizeof(num_output_channels), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of kernels")
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &num_input_channels, sizeof(num_input_channels), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of kernels")
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel height")
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel width")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Num output channels: " << num_output_channels << ", num input channels: " <<
                            num_input_channels << ", height: " << kernel_height << ", width: " <<
                            kernel_width << ", kernel_idx: " << kernel_idx)

                if(kernel_height != 0 && kernel_width != 0 && num_output_channels != 0 && num_input_channels != 0) {
                    uint32_t kernel;
                    auto *values = new fp_t[kernel_height*kernel_width]();

                    for (uint32_t out_ch = 0; out_ch < num_output_channels; out_ch++) {
                        for (uint32_t in_ch = 0; in_ch < num_input_channels; in_ch++) {
                            if(fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file) != (kernel_height*kernel_width)) {
                                PRINT_ERROR("ERROR while reading kernel values.")
                                free(values);
                                fclose(binary_file);
                                return 1;
                            }
                            std::memcpy(kernels[kernel_idx]->get_ptr_to_channel(out_ch, in_ch),
                                        values, kernel_height*kernel_width*sizeof(fp_t));
                        }
                    }

                    kernel_idx++;

                    delete[] values;
                }

                uint32_t num_biases = 0;
                if (fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of biases")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of biases: " << num_biases)

                if (num_biases) {
                    auto *bias_values = new fp_t[num_biases]();

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        PRINT_ERROR("ERROR while reading bias values.")
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(biases[bias_idx]->get_ptr_to_channel(0), bias_values, num_biases*sizeof(fp_t));

                    bias_idx++;

                    delete[] bias_values;
                }

            } else if (strcmp(buffer_layer_type, "BatchNormalization") == 0) {
                // read gamma values
                uint32_t num_gamma = 0;
                if(fread((void *) &num_gamma, sizeof(num_gamma), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of gamma values")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of gamma values: " << num_gamma)

                if(num_gamma) {
                    auto* gamma_values = new fp_t[num_gamma]();

                    if(fread((void *) gamma_values, sizeof(float), num_gamma, binary_file) != num_gamma) {
                        PRINT_ERROR("ERROR while reading gamma values.")
                        free(gamma_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(biases[bias_idx]->get_ptr_to_channel(0), gamma_values, num_gamma*sizeof(fp_t));

                    bias_idx++;

                    delete[] gamma_values;
                }

                // read beta values
                uint32_t num_beta = 0;
                if(fread((void *) &num_beta, sizeof(num_beta), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of beta values\n")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of beta values: " << num_beta)

                if(num_beta) {
                    auto *beta_values = new fp_t[num_beta]();

                    if(fread((void *) beta_values, sizeof(float), num_beta, binary_file) != num_beta) {
                        PRINT_ERROR("ERROR while reading beta values.")
                        free(beta_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(biases[bias_idx]->get_ptr_to_channel(0), beta_values, num_beta*sizeof(fp_t));

                    bias_idx++;

                    delete[] beta_values;
                }

                // read mean values
                uint32_t num_mean = 0;
                if(fread((void *) &num_mean, sizeof(num_mean), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of beta values")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of mean values: " << num_mean)

                if(num_mean) {
                    auto *mean_values = new fp_t[num_mean]();

                    if(fread((void *) mean_values, sizeof(float), num_mean, binary_file) != num_mean) {
                        PRINT_ERROR("ERROR while reading mean values.")
                        free(mean_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(biases[bias_idx]->get_ptr_to_channel(0), mean_values, num_mean*sizeof(fp_t));

                    bias_idx++;

                    delete[] mean_values;
                }

                // read variance values
                uint32_t num_variance = 0;
                if(fread((void *) &num_variance, sizeof(num_variance), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of variance values\n")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of variance values: " << num_variance)

                if(num_variance) {
                    auto *variance_values = new fp_t[num_variance]();

                    if(fread((void *) variance_values, sizeof(float), num_variance, binary_file) != num_variance) {
                        PRINT_ERROR("ERROR while reading variance values.")
                        free(variance_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(biases[bias_idx]->get_ptr_to_channel(0), variance_values, num_variance*sizeof(fp_t));

                    bias_idx++;

                    delete[] variance_values;
                }

            } else if (strcmp(buffer_layer_type, "Gemm") == 0 ||
                       strcmp(buffer_layer_type, "MatMul") == 0 ||
                       strcmp(buffer_layer_type, "Transpose") == 0) {

                uint32_t num_kernels = 0;
                uint32_t kernel_height = 0;
                uint32_t kernel_width = 0;

                if(fread((void *) &num_kernels, sizeof(num_kernels), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of kernels")
                    fclose(binary_file);
                    return 1;
                }
                if(fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel height")
                    fclose(binary_file);
                    return 1;
                }
                if(fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel width")
                    fclose(binary_file);
                    return 1;
                }

                PRINT_DEBUG("Num kernels: " << num_kernels << ", height: " << kernel_height << ", width: " << kernel_width << ", kernel_idx: " << kernel_idx)

                uint32_t kernel;
                auto *values = new fp_t[kernel_height*kernel_width]();

                for(kernel = 0; kernel < num_kernels; kernel++) {
                    if(fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file) != (kernel_height*kernel_width)) {
                        PRINT_ERROR("ERROR while reading kernel values.")
                        free(values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(kernels[kernel_idx]->get_ptr_to_channel(kernel), values, kernel_height*kernel_width*sizeof(fp_t));
                }

                kernel_idx++;

                delete[] values;

                uint32_t num_biases = 0;
                if(fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of biases")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of biases: " << num_biases)

                if(num_biases) {
                    auto *bias_values = new fp_t[num_biases]();

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        PRINT_ERROR("ERROR while reading bias values.")
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(biases[bias_idx]->get_ptr_to_channel(0), bias_values, num_biases*sizeof(fp_t));

                    bias_idx++;

                    delete[] bias_values;
                }
            } else if (strcmp(buffer_layer_type, "Add") == 0) {
                uint32_t num_biases = 0;
                if(fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of biases")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of biases: " << num_biases)

                if (num_biases) {
                    auto *bias_values = new fp_t[num_biases]();

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        PRINT_ERROR("ERROR while reading bias values.")
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy(biases[bias_idx]->get_ptr_to_channel(0), bias_values, num_biases*sizeof(fp_t));

                    bias_idx++;

                    delete[] bias_values;
                }
            } else {
                PRINT_ERROR("ERROR: Unknown layer type \"" << buffer_layer_type << "\" in weights file. Layer number: " << layer)
                fclose(binary_file);
                return 1;
            }

        }
        PRINT_DEBUG("Layer idx: " << layer << ", kernel idx: " << kernel_idx << ", bias idx: " << bias_idx)

        // Read end marker
        char end_marker[5];
        if(fread((void*)&end_marker, 1, 4, binary_file) != 4) {
            PRINT_ERROR("ERROR reading end marker")
            fclose(binary_file);
            return 1;
        }
        end_marker[4] = '\0';

        if(strcmp(end_marker,"end\n") != 0) {
            PRINT_ERROR("ERROR: Wrong end marker read: " << end_marker)
            fclose(binary_file);
            return 1;
        }
        PRINT_DEBUG(end_marker)

    }
    fclose(binary_file);
    return 0;

}
