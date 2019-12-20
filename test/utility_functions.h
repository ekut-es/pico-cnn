#include "pico-cnn/parameters.h"

#include <math.h>
#include <stdio.h>

int floatsAlmostEqual(const fp_t f1, const fp_t f2, const fp_t err);

int compare1dFloatArray(const fp_t* values, const fp_t* expected_values, const int width, fp_t error);

int compare2dFloatArray(const fp_t** values, const fp_t** expected_values,
                        const int height, const int width, fp_t error);

int compare1dIntArray(const int* values, const int* expected_values, const int width);

void print1dFloatArray_2d(const fp_t* array, const int height, const int width);
