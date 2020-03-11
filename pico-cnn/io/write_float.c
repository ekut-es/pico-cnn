#include "write_float.h"

int32_t write_float(const fp_t* image, const uint16_t height, const uint16_t width, const char* float_path) {

    uint16_t row, column;

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
