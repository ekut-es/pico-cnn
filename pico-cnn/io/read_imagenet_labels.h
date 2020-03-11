/**
 * @brief provides function to read the ImageNet labels
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef READ_IMAGENET_LABELS_H
#define READ_IMAGENET_LABELS_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


/**
 * @brief reads ImageNet labels from a given file
 *
 * @param path_to_imagenet_label
 * @param imagenet_labels will be allocated inside
 * @param num_labels number of labels which should be read from file
 *
 * @return 0 = error occured
 */
int32_t read_imagenet_labels(const char* path_to_imagenet_labels, char*** imagenet_labels, uint32_t num_labels);

#endif // READ_IMAGENET_LABELS_H
