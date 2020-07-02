/**
 * @brief provides function to read the ImageNet labels
 *
 * @author Konstantin Luebeck, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef PICO_CNN_READ_IMAGENET_LABELS_H
#define PICO_CNN_READ_IMAGENET_LABELS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

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

#endif //PICO_CNN_READ_IMAGENET_LABELS_H
