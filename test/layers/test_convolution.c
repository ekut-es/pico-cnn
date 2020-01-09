#include "test_convolution.h"

int convolution1d_output_size(int input_size, int kernel_size, int stride, int padding) {
    return (input_size - kernel_size + 2*padding)/stride  + 1;
}

void convolution2d_output_size(int input_width,int input_height, int kernel_size, int stride, int padding,
 int* output) {
    output[0] = convolution1d_output_size(input_width, kernel_size, stride, padding); // TODO: why is this correct?
    output[1] = convolution1d_output_size(input_height, kernel_size, stride, padding);
}

int test_convolution1d_naive() {

    printf("test_convolution_1d_naive()\n");
    int return_value = 0;

    #define input_size 11
    #define kernel_size 3
    #define expected_output_size 5
    fp_t error = 0.001;
    int stride =  2;
    int padding =  0;
    fp_t bias = -2;
    fp_t kernel[kernel_size] = {2, -1, 0};
    fp_t input[input_size] = {-9, 0, 3, -5, -10, 7, 3, -2, -2, 3,4};
    fp_t expected_output[expected_output_size] = {-20, 9, -29, 6, -9};

    assert(convolution1d_output_size(input_size, kernel_size, stride, padding) == expected_output_size);
    fp_t* output = (fp_t*) malloc(expected_output_size * sizeof(fp_t));

    convolution1d_naive(input, input_size, output, kernel, kernel_size, stride, padding, bias);

    return_value = compare1dFloatArray(output, expected_output, expected_output_size, error);

    free(output);

    return return_value;

    #undef input_size
    #undef kernel_size
    #undef expected_output_size
}

int test_convolution2d_naive(){

    printf("test_convolution_2d_naive()\n");
    int return_value = 0;

    return_value += test_convolution2d_naive_1();
    return_value += test_convolution2d_naive_2();

    return return_value;
}

int test_convolution2d_naive_1() {
    printf("Test 1\n");
    int return_value = 0;

    #define input_height 4
    #define input_width 3
    #define kernel_size 3
    #define expected_output_height 4
    #define expected_output_width 3

    fp_t error = 0.001;
    int stride = 1 ;
    int padding = 1;
    fp_t bias = 0;
    fp_t kernel[kernel_size*kernel_size] = {-5, -1, -3,
                                             10, 3,  0,
                                             4, -6,  6};
    fp_t input[input_height * input_width] =
     {-5,-4, -5,
      -6, 5, -10,
       8, 9, -3,
       0,-9, -1};
    fp_t expected_output[expected_output_height * expected_output_width] =
      {51,-176,  25,
        5, -41,  99,
      -39, 210,  36,
      -35, -67,-135};

    int* dimensions = (int*) malloc(2 * sizeof(int));
    convolution2d_output_size(input_width,input_height, kernel_size, stride, padding,dimensions);

    assert(dimensions[0] == expected_output_width);
    assert(dimensions[1] == expected_output_height);

    fp_t* output = (fp_t*) malloc(expected_output_width*expected_output_height * sizeof(fp_t));
    convolution2d_naive(input, input_height, input_width, output, kernel, kernel_size, stride, padding, bias);

    return_value = compare1dFloatArray(output, expected_output,expected_output_height * expected_output_width,error);

    free(dimensions);
    free(output);

    return return_value;

    #undef input_height
    #undef input_width
    #undef kernel_size
    #undef expected_output_width
    #undef expected_output_height


}

int test_convolution2d_naive_2(){

    printf("Test 2\n");
    int return_value = 0;

    #define input_height 4
    #define input_width 3
    #define kernel_size 3
    #define expected_output_height 4
    #define expected_output_width 3

    fp_t error = 0.001;
    int stride = 1 ;
    int padding = 1;
    fp_t bias = 0;
    fp_t kernel[kernel_size*kernel_size] = {0,0,0,
                                            0,0,0,
                                            0,0,0};
    fp_t input[input_height * input_width] =
     {-5,-4, -5,
      -6, 5, -10,
       8, 9, -3,
       0,-9, -1};
    fp_t expected_output[expected_output_height * expected_output_width] =
      {0,0,0,
       0,0,0,
       0,0,0,
       0,0,0};

    int* dimensions = (int*) malloc(2 * sizeof(int));
    convolution2d_output_size(input_width,input_height, kernel_size, stride, padding,dimensions);

    assert(dimensions[0] == expected_output_width);
    assert(dimensions[1] == expected_output_height);

    fp_t* output = (fp_t*) malloc(expected_output_width*expected_output_height * sizeof(fp_t));
    convolution2d_naive(input, input_height, input_width, output, kernel, kernel_size, stride, padding, bias);

    return_value = compare1dFloatArray(output, expected_output,expected_output_height * expected_output_width,error);

    free(dimensions);
    free(output);

    return return_value;

    #undef input_height
    #undef input_width
    #undef kernel_size
    #undef expected_output_width
    #undef expected_output_height

}

int test_add_channel2d_naive() {

    printf("test_add_channel2d_naive()\n");
    int return_value = 0;

    #define input_width 4
    #define input_height 3
    fp_t error = 0.001;

    fp_t input1[input_height*input_width] =
     {-6,-4, 3, -5,
       3, 4, 2,  8,
      -5, 6, 1, -4};
    fp_t input2[input_height*input_width] =
     {1, -3, -1,  5,
     -7,  7,  6, -5,
      7, -5,  4, -6};
    fp_t expected_output[input_height*input_width] =
     {-5, -7, 2,  0,
      -4, 11, 8,  3,
       2,  1, 5,-10};

    add_channel2d_naive(input1, input2, input_height, input_width);

    return_value = compare1dFloatArray(input1, expected_output, input_height * input_width, error);

    return return_value;

    #undef input_width
    #undef input_height
}
