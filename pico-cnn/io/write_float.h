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
int write_float(const float_t* image, const uint16_t height, const uint16_t width, const char* float_path) {

    int row, column;

    FILE* float_file; 
    float_file = fopen(float_path, "w+");

    if(!float_file) {
        return 1;
    }

    // magic_number
    fprintf(float_file, "FF\n");

    // dimensions
    fprintf(float_file, "%u\n", height);
    fprintf(float_file, "%u\n", width);
    
    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            fprintf(float_file, "%a\n", image[row*width+column]);
        }
    }
    
    fclose(float_file);

    return 0;
}

#endif // WRITE_FLOAT_H
