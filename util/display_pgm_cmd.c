/** 
 * @brief simple program which displays a PGM file on the command line 
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#include "pico-cnn/pico-cnn.h"
#include <stdio.h>

int main(int argc, char** argv) {
    if(argc != 4) {
        fprintf(stderr, "no path to pgm file/height/width provided!\n");
        return 1;
    }
 
    float_t* image; 
    int height = atoi(argv[2]);
    int width = atoi(argv[3]);

    if(read_pgm(&image, argv[1], 0, 0.0, 255.0) != 0) {
        fprintf(stderr, "could not read pgm file '%s'\n", argv[1]);
        return 1;
    }

    int row;
    int column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {
            printf("%02x ", (int) image[row*width+column]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
