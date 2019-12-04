#include "test_activation_function.h"


short floatsAlmostEqual(fp_t f1,fp_t f2,fp_t err){
  return abs(f1-f2) < err;
}

int test_relu_naive() {

    printf("test_relu_naive()\n");
    int return_value = 0;

    #define input_width 10
    #define input_height 1
    #define expected_output_height 1
    #define expected_output_width 10

    fp_t input[input_width] = {9, 10, -4, -5, -9, -4, -7, 5, 0, 7};
    fp_t expected_output[expected_output_width] = {9,10,0,0,0,0,0,5,0,7};

    assert(input_width == expected_output_width);

    fp_t* output = malloc(expected_output_width * sizeof(float));

    relu_naive(input, input_height, input_width, output);

    for(int i = 0; i < expected_output_width; i++) {
        if(output[i] != expected_output[i]) {
            printf("Expected: %f, Output: %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef expected_output_width
    #undef expected_output_height
}


int test_leaky_relu_naive() {

    printf("test_leaky_relu_naive()\n");
    int return_value = 0;

    #define input_width 11
    #define input_height 1
    #define expected_output_height 1
    #define expected_output_width 11
    fp_t leak =  0.01;
    fp_t error = 0.00000001;

    fp_t input[input_width] =                      {-7.8, 0.3, -9.9, 6.8, -9.4, 3.6, -9.2, 8.3, 4.3, -1.4, 0.0};
    fp_t expected_output[expected_output_width] = {-0.078, 0.3, -0.099, 6.8, -0.094, 3.6, -0.092, 8.3, 4.3, -0.014, 0.0};

    assert(input_width == expected_output_width);

    fp_t* output = malloc(expected_output_width * sizeof(float));

    leaky_relu_naive(input, input_height, input_width, output, leak);

    for(int i = 0; i < expected_output_width; i++) {
        if(!floatsAlmostEqual(output[i], expected_output[i], error)) {
            printf("Expected: %f, Output: %f\n", expected_output[i], output[i]);
            return_value = 1;
        }

    }
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef expected_output_width
    #undef expected_output_height

}

int test_parametrized_relu_naive() {

    printf("test_parametrized_relu_naive()\n");
    int return_value = 0;

    #define input_width 10
    #define input_height 1
    #define slope_width 10
    #define slope_height 1
    #define expected_output_width 10
    #define expected_output_height 1
    fp_t error = 0.00000001;

    fp_t input[input_width] =                     {-6.4, 1.9, 1.5, -6.9, -2.0, -0.3, 8.1, -2.6, 0.1, -3.8};
    fp_t slope[slope_width] =                     {0.3, 0.9, 0.8, 0.2, 0.6, 0.6, 0.7, 0.7, 0.8, 0.0};
    fp_t expected_output[expected_output_width] = {-1.92, 1.9, 1.5, -1.38, -1.2, -0.21, 8.1, -1.81, 0.1, 0.0};

    assert(input_width == expected_output_width);
    assert(input_width == slope_width);

    fp_t* output = malloc(expected_output_width * sizeof(float));

    parametrized_relu_naive(input, input_height, input_width, output, slope);

    for(int i = 0; i < expected_output_width; i++) {
        if(!floatsAlmostEqual(output[i], expected_output[i], error)) {
            printf("Expected: %f, Output: %f\n", expected_output[i], output[i]);
            return_value = 1;
        }

    }
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef slope_width
    #undef slope_height
    #undef expected_output_width
    #undef expected_output_height

}

int test_sigmoid_naive() {

    printf("tets sigmoid_naive()\n");

    int return_value = 0;

    #define input_width 10
    #define input_height 1
    #define expected_output_width 10
    #define expected_output_height 1
    fp_t error = 0.001;

    fp_t input[input_width] = {-5.4, 5.8, -4.1, 7.4, 0, 3.4, -4.4, -1.8, 2.2, -0.1};
    fp_t expected_output[expected_output_width] = {0.9955, 0.9969, 0.0163, 0.99939, 0.5, 0.9677, 0.0121, 0.1418, 0.9002, 0.4750};

    assert(input_width == expected_output_width);

    fp_t* output = malloc(expected_output_width * sizeof(float));

    sigmoid_naive(input, input_height, input_width, output);

    for(int i = 0; i < expected_output_width; i++) {
        if(!floatsAlmostEqual(output[i], expected_output[i], error)) {
            printf("Expected: %f, Output: %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef expected_output_width
    #undef expected_output_height
}
