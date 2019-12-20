#include "utility_functions.h"

int floatsAlmostEqual(fp_t f1,fp_t f2,fp_t err){
  return fabs(f1-f2) < err;
}

int compare1dFloatArray(const fp_t* values, const fp_t* expected_values, const int width,const fp_t error) {
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

int compare2dFloatArray(const fp_t** values, const fp_t** expected_values,
                        const int height, const int width, const fp_t error) {
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
int compare1dIntArray(const int* values,const int* expected_values,const int width) {
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

// TODO: check whether height and width need to be swapped
void print1dFloatArray_2d(const fp_t* array,const int height,const int width) {
    int i,j;
    for(i = 0; i < height; i++, printf("\n")){
        for(j = 0; j < width; j++) {
            printf("%f ", array[i*width + j]);
        }
    }
}
