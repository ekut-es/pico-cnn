#include "parameters.h"
#include <stdint.h>
#include <stdio.h>

/**
 * @brief makes a pgm image from an image (array)
 * 
 * @param height
 * @param width
 * @param image array which contains the image data
 * @param pgm_path full path to pgm image which should be written
 *
 * @return error (0 = success, 1 = error)
 */
int write_pgm(const uint16_t height, const uint16_t width, const float_t* image, const char* pgm_path) {

    // determine max gray value
    float_t max_gray_value = 0;
    int row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            if(image[row*width+column] > max_gray_value) {
                max_gray_value = image[row*width+column];
            }
        }
    }
    

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
    
    fprintf(pgm_file, "%u\n", (uint8_t) (max_gray_value*255.0f));

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            fputc((uint8_t) (image[row*width+column]*255.0f), pgm_file);
        }
    }


    fclose(pgm_file);

    return 0;
}
