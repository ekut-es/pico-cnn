/**
 * @brief writes an array of floats into a file
 *
 * @author Konstantin Luebeck, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */


#ifndef PICO_CNN_WRITE_FLOAT_H
#define PICO_CNN_WRITE_FLOAT_H

#include "../parameters.h"
#include <cstdint>
#include <cstdio>

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
int32_t write_float(const fp_t* image, const uint16_t height, const uint16_t width, const char* float_path);

#endif //PICO_CNN_WRITE_FLOAT_H
