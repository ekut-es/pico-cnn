/** 
 * @brief simple program to compare two float files with each other
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

typedef float fp_t;

void usage() {
    printf("USAGE:\n");
    printf("compare_float_files FILE_A FILE_B [TOLERANCE]\n");
}

int main(int argc, char** argv) {
    if(argc < 3) {
        printf("ERROR: no float files were provided!\n");
        usage();
        return 1;
    }
   
    fp_t tolerance = 0.0;

    if(argc == 4) {
        sscanf(argv[3], "%f", &tolerance);
    }
    

    FILE *file_a;
    file_a = fopen(argv[1], "r");

    if(!file_a) {
        printf("ERROR: could not open '%s'!\n", argv[1]);
        return 1;
    }

    FILE *file_b;
    file_b = fopen(argv[2], "r");

    if(!file_b) {
        printf("ERROR: could not open '%s'!\n", argv[2]);
        fclose(file_b);
        return 1;
    }

    char buffer_a[100];
    char buffer_b[100];

    // read magic number 
    fgets(buffer_a, 100, file_a);
    
    if(strncmp(buffer_a, "FF", 2) != 0) {
        printf("ERROR: '%s' is not a float file!\n", argv[1]);
        fclose(file_a);
        fclose(file_b);
        return 1;
    }

    fgets(buffer_b, 100, file_b);

    if(strncmp(buffer_b, "FF", 2) != 0) {
        printf("ERROR: '%s' is not a float file!\n", argv[1]);
        fclose(file_a);
        fclose(file_b);
        return 1;
    }

    // read height and width
    int height, width;

    fgets(buffer_a, 100, file_a);
    height = atoi(buffer_a);

    fgets(buffer_a, 100, file_a);
    width = atoi(buffer_a);

    fgets(buffer_b, 100, file_b);
    if(atoi(buffer_b) != height) {
        printf("ERROR: files have different heights!\n");
        printf("%d %s\n", height, argv[1]);
        printf("%d %s\n", atoi(buffer_b), argv[2]);
        fclose(file_a);
        fclose(file_b);
        return 1;
    }

    fgets(buffer_b, 100, file_b);
    if(atoi(buffer_b) != width) {
        printf("ERROR: files have different widths!\n");
        printf("%d %s\n", height, argv[1]);
        printf("%d %s\n", atoi(buffer_b), argv[2]);
        fclose(file_a);
        fclose(file_b);
        return 1;
    }

    printf("%dx%d\n", height, width);
    printf("tolerance: %.9f\n", tolerance);

    fp_t float_a, float_b;

    int row, column;

    for(row = 0; row < height; row++) {
        for(column = 0; column < width; column++) {

            fscanf(file_a, "%a", &float_a);
            fscanf(file_b, "%a", &float_b);

            if(fabsf(float_a-float_b) > tolerance) {
                printf("(%d,%d): a: %.9f, b: %.9f, a-b: %.9f\n", row, column, float_a, float_b, float_a-float_b);
            }
        }
    }

    fclose(file_a);
    fclose(file_b);

    return 0;
}
