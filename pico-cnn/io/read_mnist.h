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
 * TODO make it a macro
 */
static uint32_t change_endianess(uint32_t value);

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
int read_mnist_images(const char* path_to_mnist_images, fp_t*** mnist_images, uint32_t num_images, const uint8_t padding, const fp_t lower_bound, const fp_t upper_bound);

/**
 * @brief reads MNIST labels from a given file
 *
 * @param path_to_mnist_labels full path to file which contains MNIST labels
 * @param mnist_labels array in which the labels should be stored
 * @param num_labels number of labels which should be read from file
 *
 * @return number of labels which were read from file (0 = error)
 */
int read_mnist_labels(const char* path_to_mnist_labels, uint8_t** mnist_labels, uint32_t num_labels);

#endif // READ_MNIST_H
