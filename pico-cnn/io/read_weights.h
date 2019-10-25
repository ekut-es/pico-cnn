/**
 * @brief provides functions to read the weights running CNNs
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef READ_WEIGHTS_H
#define READ_WEIGHTS_H

//#define DEBUG

#include "../parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>



/**
 * @brief reads weights (elements of kernels and biasses) from a weight file
 * Find the definition of the file format in doc/weight_file_format.md
 *
 * @param path_to_weight_file file in which the weights are stored
 * @param kernels a 4-dimensional array which stores the kernels:
 * (*kernels)[layer_num][kernel_num][kernel_element]. Memory for it will be
 * allocated inside of the function
 * @param biasses a 3-dimensional array which stores the biasses:
 * (*biasses)[layer_num][bias_num]
 *
 * @return =! 0 means an error occured
 */
int read_weights(const char* path_to_weights_file, fp_t**** kernels, fp_t*** biasses);

#endif // READ_WEIGHTS_H
