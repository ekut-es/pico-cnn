#include "utility_functions.h"

int32_t floatsAlmostEqual(fp_t f1,fp_t f2,fp_t err){
  return fabs(f1-f2) < err;
}

// is also used to compare 2d data represented by a 1d array
int32_t compare1dFloatArray(fp_t* values, fp_t* expected_values, uint32_t width, fp_t error) {
    uint32_t i;
    int32_t return_value = 0;

    for(i = 0; i < width; i++){
        if(!floatsAlmostEqual(values[i], expected_values[i], error)) {
            printf("Error at position %d. Expected: %f, Output: %f\n", i, expected_values[i], values[i]);
            return_value = 1;
        }
    }
    return return_value;
}

int32_t compare2dFloatArray(fp_t** values, fp_t** expected_values,
                            uint32_t height, uint32_t width, fp_t error) {
    uint32_t i,j;
    int32_t return_value = 0;

    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            if(!floatsAlmostEqual(values[i][j], expected_values[i][j], error)){
                printf("Error at position (%d,%d). Expected: %f, Output; %f, \n",
                         i,j, expected_values[i][j], values[i][j]);
                return_value = 1;
            }
        }
    }

    return return_value;
}

int32_t compare1dIntArray(int32_t* values,int32_t* expected_values,uint32_t width) {
    uint32_t i;
    int32_t return_value = 0;

   for(i = 0; i < width; i++) {
       if(values[i] != expected_values[i]) {
           printf("Error at position %d. Expected: %d, Output: %d\n", i, expected_values[i], values[i]);
           return_value = 1;
       }
   }
   return return_value;
}

void print1dFloatArray_2d(fp_t* array, uint32_t height, uint32_t width) {
    uint32_t i,j;
    for(i = 0; i < height; i++, printf("\n")){
        for(j = 0; j < width; j++) {
            printf("%f ", array[i*width + j]);
        }
    }
}

void print2dFloatArray_3d(fp_t** array, uint32_t depth, uint32_t height, uint32_t width) {
    uint32_t i,j,k;
    for(i = 0; i < depth; i++,printf("\n\n")) {
        for(j = 0; j < height; j++, printf("\n")) {
            for(k = 0; k < width; k++) {
                printf("%f ", array[i][j*width + k]);
            }
        }
    }
}

void initialize2dFloatArray(fp_t* values, uint32_t num_channels, uint32_t height, uint32_t width, fp_t** channels) {
    // uint32_t channel,row,pixel;
    //
    // for(channel = 0; channel < num_channels;channel++) {
    //     for(row = 0; row < height;row++) {
    //         for(pixel= 0; pixel < width; pixel++) {
    //             channels[channel][width*row+pixel] = values[height*width*channel + width*row + pixel];
    //         }
    //     }
    // }
}
