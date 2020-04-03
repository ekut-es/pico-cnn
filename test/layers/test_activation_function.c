#include "test_activation_function.h"

int32_t test_relu_naive() {

    printf("test_relu_naive()\n");
    int32_t return_value = 0;

    #define input_width 10
    #define input_height 1
    fp_t error = 0.001;

    fp_t input[input_width] =           {9,11,-4,-5,-9,-4,-7, 5, 0, 7};
    fp_t expected_output[input_width] = {9,11, 0, 0, 0, 0, 0, 5, 0, 7};

    fp_t* output = (fp_t*) malloc(input_width * input_height * sizeof(fp_t));

    relu_naive(input, input_height, input_width, output);

    return_value = compare1dFloatArray(output, expected_output, input_width,error);
    
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
}


int32_t test_leaky_relu_naive() {

    printf("test_leaky_relu_naive()\n");
    int32_t return_value = 0;

    #define input_width 11
    #define input_height 1
    fp_t leak =  0.01;
    fp_t error = 0.0001;

    fp_t input[input_width] =            {  -7.8, 0.3,   -9.9, 6.8,   -9.4, 3.6,   -9.2, 8.3, 4.3,   -1.4, 0.0};
    fp_t expected_output[input_width] =  {-0.078, 0.3, -0.099, 6.8, -0.094, 3.6, -0.092, 8.3, 4.3, -0.014, 0.0};

    fp_t* output = malloc(input_width * sizeof(fp_t));

    leaky_relu_naive(input, input_height, input_width, output, leak);

    return_value = compare1dFloatArray(output, expected_output, input_width, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef expected_output_width
    #undef expected_output_height

}

int32_t test_parametrized_relu_naive() {

    printf("test_parametrized_relu_naive()\n");
    int32_t return_value = 0;

    #define input_width 10 // also slope width
    #define input_height 1 // also slope height
    fp_t error = 0.0001;

    fp_t input[input_width] =           { -6.4, 1.9, 1.5,  -6.9, -2.0,  -0.3, 8.1,    -2.6, 0.1, -3.8};
    fp_t slope[input_width] =           {  0.3, 0.9, 0.8,   0.2,  0.6,   0.6, 0.7,     0.7, 0.8,  0.0};
    fp_t expected_output[input_width] = {-1.92, 1.9, 1.5, -1.38, -1.2, -0.18, 8.1, -1.8199, 0.1,  0.0};

    fp_t* output = malloc(input_width * sizeof(fp_t));

    parametrized_relu_naive(input, input_height, input_width, output, slope);

    return_value = compare1dFloatArray(output, expected_output, input_width, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
}

int32_t test_sigmoid_naive() {

    printf("test_sigmoid_naive()\n");
    int32_t return_value = 0;

    #define input_width 10
    #define input_height 1
    fp_t error = 0.001;

    fp_t input[input_width] =
        {-5.4, 5.8, -4.1, 7.4, 0, 3.4, -4.4, -1.8, 2.2, -0.1};
    fp_t expected_output[input_width] =
        {0.0044962, 0.9969, 0.0163, 0.99939, 0.5, 0.9677, 0.0121, 0.1418, 0.9002, 0.4750};

    fp_t* output = malloc(input_width * sizeof(fp_t));

    sigmoid_naive(input, input_height, input_width, output);

    return_value = compare1dFloatArray(output, expected_output, input_width, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
}

int32_t test_softmax_naive() {

    printf("test_softmax_naive()\n");
    int32_t return_value = 0;

    #define input_width 10
    #define input_height 1
    fp_t error = 0.00001;

    fp_t input[input_width] =
       {0.1, -6.8, -0.4, -0.0, -2.7,
        4.5, -5.2, -5.5,  6.9, -0.2};
    fp_t expected_output[input_width] = {
        0.001010, 0.000002560,  0.000617, 0.000920,   0.00006188,
        0.082891, 0.000005079,0.00000376, 0.913727,   0.0007539};

    fp_t* output = malloc(input_width * sizeof(fp_t));

    softmax_naive(input, input_height, input_width, output);

    return_value = compare1dFloatArray(output, expected_output, input_width, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height

}

int32_t test_local_response_normalization_naive() {

    int32_t return_value = 0;
    printf("test_local_response_normalization_naive()\n");

    #define input_width 3
    #define input_height 2
    #define input_depth 3
    fp_t error = 0.0001;
    fp_t alpha = 0.1;
    fp_t beta =  0.5;
    int32_t n = 2;

    fp_t input1[input_width*input_height] = {-1, -2,  2, 0, -2, -2};
    fp_t input2[input_width*input_height] = {-1, -5, -1, 2,  1,  0};
    fp_t input3[input_width*input_height] = {-1,  1, -3, 2,  5, -4};

    fp_t* input[input_depth];
    input[0] = input1;
    input[1] = input2;
    input[2] = input3;

    fp_t expected_output1[input_width * input_height] =
      {-0.95342, -1.2777, 1.7888,
              0, -1.7888,-1.8257};
    fp_t expected_output2[input_width * input_height] =
       {-0.9325, -3.1622,-0.76694,
         1.69030 ,0.63245,      0};
    fp_t expected_output3[input_width * input_height] =
      {-0.9534625,0.659380, -2.44948,
          1.69030, 3.29690,-2.981423};

    fp_t* expected_output[input_depth];
    expected_output[0] = expected_output1;
    expected_output[1] = expected_output2;
    expected_output[2] = expected_output3;

    fp_t** output = (fp_t**) malloc(input_depth *sizeof(fp_t*));
    for(int32_t i = 0; i < input_depth; i++) {
        output[i] = (fp_t*) malloc(input_width * input_height * sizeof(fp_t));
    }

    local_response_normalization_naive((fp_t**) input, input_height, input_width, input_depth, output,
                                        alpha, beta, n);

    return_value = compare2dFloatArray(output, expected_output, input_depth, input_height * input_width,error);

    for(int32_t i = 0; i < input_depth; i++){
        free(output[i]);
    }
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef input_depth
}
