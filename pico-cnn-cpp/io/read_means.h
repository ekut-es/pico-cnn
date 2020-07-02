/**
 * @brief provides functions to read means previously generated from
 * tiny-dnn/caffe CNNs
 *
 * @author Konstantin Luebeck, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_READ_MEANS_H
#define PICO_CNN_READ_MEANS_H

#include "../parameters.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

/**
 * @brief reads three means from a means file
 * Find the definition of the file format in doc/mean_file_format.md
 *
 * @param path_to_means_file file in which the means are stored
 * @param 1D-array of fp_t (size 3) to store the means
 *
 * @return =! 0 means an error occured
 */
int32_t read_means(const char* path_to_means_file, fp_t* means);

#endif //PICO_CNN_READ_MEANS_H
