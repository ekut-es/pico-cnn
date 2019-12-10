/**
 * @brief writes an array of floats into a file
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef WRITE_FLOAT_H
#define WRITE_FLOAT_H

#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>

/**
 * @brief writes an array of floats into a file
 *
 * @param image (height x width)
 * @param height
 * @param width
 * @param float_path full file path
 *
 * @return success = 0
 */
int write_float(const fp_t* image, const uint16_t height, const uint16_t width, const char* float_path);

#endif // WRITE_FLOAT_H
