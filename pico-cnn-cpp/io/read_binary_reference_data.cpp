#include "read_binary_reference_data.h"

int32_t read_binary_reference_input_data(const char* path_to_sample_data, pico_cnn::naive::Tensor **input_tensor) {

    if ((*input_tensor)->num_dimensions() != 4) {
        PRINT_ERROR("Reference input data can only be copied into a 4D-Tensor (num_batches, num_channels, height, width).")
        return 1;
    }

    FILE *binary_file;
    binary_file = fopen(path_to_sample_data, "r");

    if(binary_file != nullptr) {

        // Read magic number
        char magic_number[5];
        if(fread((void*)&magic_number, 1, 4, binary_file) != 4) {
            PRINT_ERROR("ERROR reading magic number of input file.")
            fclose(binary_file);
            return 1;
        }
        magic_number[4] = '\0';

        if(strcmp(magic_number,"FCI\n") != 0) {
            PRINT_ERROR("ERROR: Wrong magic number: " << magic_number)
            fclose(binary_file);
            return 1;
        }
        PRINT_DEBUG(magic_number)

        // Read number of batches
        uint32_t num_batches;
        if(fread((void*)&num_batches, sizeof(num_batches), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading number of batches")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Number of batches: " << num_batches)
            if (num_batches != 1) {
                PRINT_ERROR("Number of batches must be == 1")
                fclose(binary_file);
                return 1;
            }
        }

        // Read number of channels
        uint32_t num_channels;
        if(fread((void*)&num_channels, sizeof(num_channels), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading number of channels")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Number of channels: " << num_channels)
            if (num_channels != (*input_tensor)->num_channels()) {
                PRINT_ERROR("Number of channels in binary file: " << num_channels <<
                            " does not match tensor shape: " << (*input_tensor)->num_channels())
                fclose(binary_file);
                return 1;
            }
        }

        // Read channel height
        uint32_t height;
        if(fread((void*)&height, sizeof(height), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading height")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Height: " << height)
            if (height != (*input_tensor)->height()) {
                PRINT_ERROR("Height in binary file: " << height <<
                            " does not match tensor shape: " << (*input_tensor)->height())
                fclose(binary_file);
                return 1;
            }
        }

        // Read channel width
        uint32_t width;
        if(fread((void*)&width, sizeof(width), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading width")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Width: " << width)
            if (width != (*input_tensor)->width()) {
                PRINT_ERROR("Width in binary file: " << width <<
                            " does not match tensor shape: " << (*input_tensor)->width())
                fclose(binary_file);
                return 1;
            }
        }

        for(uint32_t channel = 0; channel < num_channels; channel++) {

            auto *values = new fp_t[height*width]();

            uint32_t numbers_read = 0;

            numbers_read = fread((void*)values, sizeof(float), height*width, binary_file);

            if(numbers_read != height*width) {
                PRINT_ERROR("ERROR reading data. numbers_read = " << numbers_read)
                free(values);
                return 1;
            } else {
                #ifdef DEBUG
                for(uint32_t i = 0; i < height*width; i++) {
                    PRINT_DEBUG(values[i])
                }
                #endif
                std::memcpy((*input_tensor)->get_ptr_to_channel(0, channel), values, height*width*sizeof(fp_t));
            }

            delete[] values;
        }

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

int32_t read_binary_reference_output_data(const char* path_to_sample_data, pico_cnn::naive::Tensor **output_tensor) {

    if ((*output_tensor)->num_dimensions() != 2) {
        PRINT_ERROR("Reference ouptut data can only be copied into a 1D-Tensor (num_outputs).")
        return 1;
    }

    FILE *binary_file;
    binary_file = fopen(path_to_sample_data, "r");

    if(binary_file != nullptr) {

        // Read magic number
        char magic_number[5];
        if(fread((void*)&magic_number, 1, 4, binary_file) != 4) {
            PRINT_ERROR("ERROR reading magic number of output file.")
            fclose(binary_file);
            return 1;
        }
        magic_number[4] = '\0';

        if(strcmp(magic_number,"FCO\n") != 0) {
            PRINT_ERROR("ERROR: Wrong magic number: " << magic_number)
            fclose(binary_file);
            return 1;
        }
        PRINT_DEBUG(magic_number)

        // Read number of batches
        uint32_t num_batches;
        if(fread((void*)&num_batches, sizeof(num_batches), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading number of batches")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Number of batches: " << num_batches)
            if (num_batches != 1) {
                PRINT_ERROR("Number of batches must be == 1")
                fclose(binary_file);
                return 1;
            }
        }

        // Read number of outputs
        uint32_t num_outputs;
        if(fread((void*)&num_outputs, sizeof(num_outputs), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading number of outputs")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Number of outputs: " << num_outputs)
            if (num_outputs != (*output_tensor)->num_elements()) {
                PRINT_ERROR("Number of outputs in binary file: " << num_outputs << " and output tensor shape: " << (*output_tensor)->num_elements() << " do not match.")
                fclose(binary_file);
                return 1;
            }
        }

        auto *values = new fp_t[num_outputs]();

        if(fread((void*)values, sizeof(float), num_outputs, binary_file) != num_outputs) {
            PRINT_ERROR("ERROR reading output values.")
            free(values);
            fclose(binary_file);
            return 1;
        }

        #ifdef DEBUG
        for(uint32_t i = 0; i < num_outputs; i++) {
            PRINT_DEBUG(values[i])
        }
        #endif

        std::memcpy((*output_tensor)->get_ptr_to_channel(0, 0), values, num_outputs*sizeof(fp_t));

        delete[] values;

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
