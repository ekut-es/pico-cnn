/**
 * @brief writes an array of floats into a pgm file
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef WRITE_PGM_H
#define WRITE_PGM_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

/**
 * @brief creates an pgm file from an image (array)
 *
 * @param image array which contains the image data
 * @param height
 * @param width
 * @param pgm_path full path to pgm image which should be written
 *
 * @return error (0 = success, 1 = error)
 */
int32_t write_pgm(const fp_t* image, const uint16_t height, const uint16_t width, const char* pgm_path);

#endif // WRITE_PGM_H
