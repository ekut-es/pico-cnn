#include "write_pgm.h"

int32_t write_pgm(const fp_t* image, const uint16_t height, const uint16_t width, const char* pgm_path) {

    // determine max gray value
    fp_t max_gray_value = -1000;
    fp_t min_gray_value = 1000;
    uint16_t row, column;

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

    fp_t gray_value_range = fabsf(min_gray_value - max_gray_value);

    FILE* pgm_file;
    pgm_file = fopen(pgm_path, "w+");

    if(!pgm_file) {
        return 1;
    }

    // magic_number
    fprintf(pgm_file, "P5\n");
    // comment
    fprintf(pgm_file, "# pico-cnn\n");
    // write dimensions
    fprintf(pgm_file, "%u %u\n", width, height);
    // write max gray value
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
