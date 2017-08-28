/** 
 * @brief provides functions to read the MNIST training/testing images and labels
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef READ_MNIST_H
#define READ_MNIST_H

#include "../parameters.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief changes the endianess of a 32-bit integer
 *
 * @param value input
 *
 * @return input with changed endianess
 */
static uint32_t change_endianess(uint32_t value) {
    uint32_t result = 0;
    result |= (value & 0x000000FF) << 24;
    result |= (value & 0x0000FF00) << 8;
    result |= (value & 0x00FF0000) >> 8;
    result |= (value & 0xFF000000) >> 24;
    return result;
}

/**
 * @brief reads MNIST images from a given file
 *
 * @param path_to_mnist_images full path to file which contains MNIST images
 * @param mnist_images 2D array in which the images should be stored [image number][image content]
 * @param num_images number of images which should be read from file
 * @param padding number of 0 pixels which should be added to the border of
 * each image
 * @param lower_bound to which float range should the images be converted
 * [lower_bound, upper_bound]
 * @param upper_bound to which float range should the images be converted
 * [lower_bound, upper_bound]
 *
 * @return number of images which were read from file (0 = error)
 */
int read_mnist_images(const char* path_to_mnist_images, float_t*** mnist_images, uint32_t num_images, const uint8_t padding, const float_t lower_bound, const float_t upper_bound) {

    FILE *images;
    images = fopen(path_to_mnist_images, "r");

    if(images != 0) {
        uint32_t magic_number;
        uint32_t num_images_provided;
        uint32_t height;
        uint32_t width;
        float_t range = fabs(lower_bound - upper_bound);

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
        *mnist_images = (float_t**) malloc(num_images*sizeof(float_t*));

        // start reading images
        for(i = 0; i < num_images; i++) {
            
            // allocate memory for single image
            (*mnist_images)[i] = (float_t*) malloc((width+2*padding)*(height+2*padding)*sizeof(float_t));

            uint32_t row, column;
            uint8_t buffer[width*height];
            // store single image
            fread(&buffer, 1, height*width, images);

            for(row = 0; row < height+2*padding; row++) {
                for(column = 0; column < width+2*padding; column++) {
                    if(row < padding || row >= height+padding) {
                        (*mnist_images)[i][row*(width+2*padding)+column] = lower_bound;
                    } else if(column < padding || column >= width+padding) {
                        (*mnist_images)[i][row*(width+2*padding)+column] = lower_bound;
                    } else {
                        (*mnist_images)[i][row*(width+2*padding)+column] = (((buffer[(row-padding)*width+column-padding] / 255.0f) * range) + lower_bound);
                    }
                }
            }
        }

        fclose(images);

        return num_images;
    }

    return 0;
}

/**
 * @brief reads MNIST labels from a given file
 *
 * @param path_to_mnist_labels full path to file which contains MNIST labels
 * @param mnist_labels array in which the labels should be stored
 * @param num_labels number of labels which should be read from file
 *
 * @return number of labels which were read from file (0 = error)
 */
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

#endif // READ_MNIST_H
