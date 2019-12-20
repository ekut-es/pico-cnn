#include "test_fully_connected.h"

int test_fully_connected() {

    printf("test_fully_connected()\n");
    int return_value = 0;

    #define input_width 6
    #define output_width 4
    const fp_t error = 0.01;

    const fp_t input[input_width] = {-2, 4, 1, 8, -5, 0};
    const fp_t kernel[input_width * output_width] =
      {-0.5,  0.2, -0.3,  0.1,
       -0.1, -1.0, -0.9,  0.4,
        0.8,  0.5, -0.1, -0.4,
       -0.4, -0.9, -0.9, -0.8,
       -0.3,  0.5,  0.7, -0.9,
       -0.3,  0.4,  0.9, -0.7};
    const fp_t bias[output_width] = {6,-3,0,1}  ;
    const fp_t expected_output[output_width] = {5.6999,-16.6,-13.8,0.0999};

    fp_t* output = (fp_t*) malloc(output_width * sizeof(fp_t));

    fully_connected_naive(input, input_width, output, output_width, kernel, bias);

    // for(int i = 0; i < output_width; i++) {
    //     if(1||!floatsAlmostEqual(output[i],expected_output[i],error)) {
    //         printf("Expected: %f, Output: %f\n", expected_output[i], output[i]);
    //         return_value = 1;
    //     }
    // }
    return_value = compare1dFloatArray(output, expected_output, output_width, error);
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
}
