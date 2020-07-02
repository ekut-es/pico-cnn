/**
 * @brief reads a pgm file and stores it into a 1-dimensional array
 *
 * @author Konstantin Luebeck, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */


#ifndef PICO_CNN_READ_PGM_H
#define PICO_CNN_READ_PGM_H

#include "../parameters.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

/**
 * @brief reads a pgm file and stores it into a 1-dimensional array
 * dimensions of the read image should be known before
 *
 * @param image array which contains the image data (will be allocated inside)
 * @param pgm_path full path to pgm image which should be read
 * @param padding which should be added to the edges (lower_bound value)
 * @param lower_bound of range to which a pixel should be scaled
 * @param upper_bound of range to which a pixel should be scaled
 *
 * @return error (0 = success, 1 = error)
 */
int32_t read_pgm(fp_t** image, const char* pgm_path, const uint8_t padding, const fp_t lower_bound, const fp_t upper_bound, uint16_t* height, uint16_t* width);

#endif //PICO_CNN_READ_PGM_H
