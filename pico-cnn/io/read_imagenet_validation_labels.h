/**
 * @brief provides function to read the ImageNet validation labels
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef READ_IMAGENET_VALIDATION_LABELS_H
#define READ_IMAGENET_VALIDATION_LABELS_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FILE_NAME_LENGTH 28

/**
 * @brief reads ImageNet labels from a given file
 *
 * assert: list of file names is sorted 0 -> 50000
 * assert: length of file name is constant FILE_NAME_LENGTH
 * @param path_to_imagenet_label
 * @param imagenet_validation_labels will be allocated inside [image_number] = class
 * @param num_labels number of labels which should be read from file
 *
 * @return 0 = error occured
 */
int32_t read_imagenet_validation_labels(const char* path_to_imagenet_validation_labels, uint32_t** imagenet_validation_labels, uint32_t num_labels);

#endif // READ_IMAGENET_VALIDATION_LABELS_H
