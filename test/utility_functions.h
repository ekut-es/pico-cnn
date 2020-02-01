#include "pico-cnn/parameters.h"

#include <math.h>
#include <stdio.h>

int floatsAlmostEqual(fp_t f1, fp_t f2, fp_t err);

int compare1dFloatArray(fp_t* values, fp_t* expected_values, int width, fp_t error);

int compare2dFloatArray(fp_t** values, fp_t** expected_values,
                        int height, int width, fp_t error);

int compare1dIntArray(int* values, int* expected_values, int width);

void print1dFloatArray_2d(fp_t* array, int height, int width);

void print2dFloatArray_3d(fp_t** array, int depth, int height, int width);

// write values from 1d array into 2d array
void initialize2dFloatArray(fp_t* values, int num_channels, int height, int width, fp_t** channels); 
