/* Little C header to read the MNIST dataset.
 *
 * Author: Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

uint32_t change_endianess(uint32_t value) {
    uint32_t result = 0;
    result |= (value & 0x000000FF) << 24;
    result |= (value & 0x0000FF00) << 8;
    result |= (value & 0x00FF0000) >> 8;
    result |= (value & 0xFF000000) >> 24;
    return result;
}

// TODO uint8_t to float
// returns number of images read
// 0 : problem occured
int read_mnist_images(const char* path_to_mnist_images, uint8_t*** mnist_images, uint32_t num_images, uint8_t padding) {

    FILE *images;
    images = fopen(path_to_mnist_images, "r");

    if(images != 0) {
        uint32_t magic_number;
        uint32_t num_images_provided;
        uint32_t height;
        uint32_t width;

        int i;

        fread(&magic_number, sizeof(magic_number), 1, images);
        magic_number = change_endianess(magic_number);

        // check magic number
        if(magic_number != 2051) {
            fclose(images);
            return 0;
        }

		fread(&num_images_provided, sizeof(num_images_provided), 1, images);
        num_images_provided = change_endianess(num_images_provided);

		fread(&height, sizeof(height), 1, images);
        height = change_endianess(height);

		fread(&width, sizeof(width), 1, images);
        width = change_endianess(width);

        // if there are less images in the dataset than requested reduce
        // the number of requested images to number of images provided
        if(num_images_provided < num_images) {
            num_images = num_images_provided;
        }

        // allocate pointers for images
        *mnist_images = (uint8_t**) malloc(num_images*sizeof(uint8_t*));

        // start reading images
        for(i = 0; i < num_images; i++) {
            
            // allocate memory for single image
            (*mnist_images)[i] = (uint8_t*) malloc((width+2*padding)*(height+2*padding)*sizeof(uint8_t));
            // store single image
            if(padding == 0) {
			    fread(&(*mnist_images)[i][0], 1, height*width, images);
            } else {
                uint8_t buffer[width*height];
			    fread(&buffer, 1, height*width, images);

                uint32_t row, column;
                for(row = 0; row < height+2*padding; row++) {
                    for(column = 0; column < width+2*padding; column++) {
                        if(row < padding || row >= height+padding) {
                            (*mnist_images)[i][row*(width+2*padding)+column] = 0;
                        } else if(column < padding || column >= width+padding) {
                            (*mnist_images)[i][row*(width+2*padding)+column] = 0;
                        } else {
                            (*mnist_images)[i][row*(width+2*padding)+column] = buffer[(row-padding)*width+column-padding];
                        }
                    }
                }
            }
        }

        fclose(images);

        return num_images;
    }

    return 0;
}

// returns number of labels read
// 0 : problem occured
int read_mnist_labels(const char* path_to_mnist_labels, uint8_t** mnist_labels, uint32_t num_labels) {

	FILE *labels;
    labels = fopen(path_to_mnist_labels, "r");

	if(labels != 0) {
        uint32_t magic_number;
        uint32_t num_labels_provided;

        int i;

        fread(&magic_number, sizeof(magic_number), 1, labels);
        magic_number = change_endianess(magic_number);

        // check magic number
        if(magic_number != 2049) {
            fclose(labels);
            return 0;
        }

		fread(&num_labels_provided, sizeof(num_labels_provided), 1, labels);
        num_labels_provided = change_endianess(num_labels_provided);

        // if there are less labels in the dataset than requested reduce
        // the number of requested labels to number of labels provided
        if(num_labels_provided < num_labels) {
            num_labels = num_labels_provided;
        }

		// allocate pointers for all images
        *mnist_labels = (uint8_t*) malloc(num_labels*sizeof(uint8_t));
        
        // start reading images
        for(i = 0; i < num_labels; i++) {
			uint8_t label = getc(labels);
        	(*mnist_labels)[i] = label;
        }

        fclose(labels);

        return num_labels;
    }

    return 0;
}
