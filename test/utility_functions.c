#include "utility_functions.h"

int floatsAlmostEqual(fp_t f1,fp_t f2,fp_t err){
  return fabs(f1-f2) < err;
}

int compare1dFloatArray(fp_t* values, fp_t* expected_values, int width, fp_t error) {
    int i;
    int return_value = 0;

    for(i = 0; i < width; i++){
        if(!floatsAlmostEqual(values[i], expected_values[i], error)) {
            printf("Error at position %d. Expected: %f, Output: %f\n", i, expected_values[i], values[i]);
            return_value = 1;
        }
    }
    return return_value;
}

int compare2dFloatArray(fp_t** values, fp_t** expected_values,
                        int height, int width, fp_t error) {
    int i,j;
    int return_value = 0;

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

int compare1dIntArray(int* values,int* expected_values,int width) {
   int i;
   int return_value = 0;

   for(i = 0; i < width; i++) {
       if(values[i] != expected_values[i]) {
           printf("Error at position %d. Expected: %d, Output: %d\n", i, expected_values[i], values[i]);
           return_value = 1;
       }
   }
   return return_value;
}

void print1dFloatArray_2d(fp_t* array,int height,int width) {
    int i,j;
    for(i = 0; i < height; i++, printf("\n")){
        for(j = 0; j < width; j++) {
            printf("%f ", array[i*width + j]);
        }
    }
}

void print2dFloatArray_3d(fp_t** array, int depth, int height, int width) {
    int i,j,k;
    for(i = 0; i < depth; i++,printf("\n\n")) {
        for(j = 0; j < height; j++, printf("\n")) {
            for(k = 0; k < width; k++) {
                printf("%f ", array[i][j*width + k]);
            }
        }
    }
}
