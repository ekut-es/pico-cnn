/** 
 * @brief writes an array of floats into a pgm file
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */


#include "../parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

/**
 * @brief creates an pgm file from an image (array)
 * 
 * @param height
 * @param width
 * @param image array which contains the image data
 * @param pgm_path full path to pgm image which should be written
 *
 * @return error (0 = success, 1 = error)
 */
int write_pgm(const float_t* image, const uint16_t height, const uint16_t width, const char* pgm_path) {

    // determine max gray value
    float_t max_gray_value = -1000;
    float_t min_gray_value = 1000;
    int row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            if(image[row*width+column] > max_gray_value) {
                max_gray_value = image[row*width+column];
            }
            if(image[row*width+column] < min_gray_value) {
                min_gray_value = image[row*width+column];
            }
        }
    }

    float_t gray_value_range = fabsf(min_gray_value - max_gray_value);

    FILE* pgm_file; 
    pgm_file = fopen(pgm_path, "w+");

    if(!pgm_file) {
        return 1;
    }

    // magic_number
    fprintf(pgm_file, "P5\n");
    // comment
    fprintf(pgm_file, "# pico-cnn\n");
    fprintf(pgm_file, "%u %u\n", width, height);
    
    fprintf(pgm_file, "%u\n", 255);

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            // normalize
            uint8_t pixel = (uint8_t) ( ( (image[row*width+column]-min_gray_value) / gray_value_range ) * 255.0f );
            fputc(pixel, pgm_file);
        }
    }


    fclose(pgm_file);

    return 0;
}
