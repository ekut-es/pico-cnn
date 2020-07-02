/**
 * @brief provides function to read the ImageNet validation labels
 *
 * @author Konstantin Luebeck, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_READ_IMAGENET_VALIDATION_LABELS_H
#define PICO_CNN_READ_IMAGENET_VALIDATION_LABELS_H

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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

#endif //PICO_CNN_READ_IMAGENET_VALIDATION_LABELS_H
