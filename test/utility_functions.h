#include "pico-cnn/parameters.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>

int32_t floatsAlmostEqual(fp_t f1, fp_t f2, fp_t err);

int32_t compare1dFloatArray(fp_t* values, fp_t* expected_values, uint32_t width, fp_t error);

int32_t compare2dFloatArray(fp_t** values, fp_t** expected_values,
                            uint32_t height, uint32_t width, fp_t error);

int32_t compare1dIntArray(int32_t* values, int32_t* expected_values, uint32_t width);

void print1dFloatArray_2d(fp_t* array, uint32_t height, uint32_t width);

void print2dFloatArray_3d(fp_t** array, uint32_t depth, uint32_t height, uint32_t width);

// write values from 1d array into 2d array
void initialize2dFloatArray(fp_t* values, uint32_t num_channels, uint32_t height, uint32_t width, fp_t** channels);
