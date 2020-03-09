#include "test_batch_normalization.h"

int32_t test_batch_normalization_naive_1() {

    printf("test_batch_normalization_naive_1()\n");
    int32_t return_value = 0;

    #define input_width 3
    #define input_height 1
    fp_t error = 0.001;

    fp_t gamma, beta, mean, variance, epsilon;

    gamma = 1.0;
    beta = 0;
    mean = 0;
    variance = 1;
    epsilon = 1e-5;

    fp_t input[input_height*input_width] =           {-1, 0, 1};
    fp_t expected_output[input_height*input_width] = {-0.999995, 0.0, 0.999995};

    fp_t* output = (fp_t*) malloc(input_height * input_width * sizeof(fp_t));

    batch_normalization_naive(input, input_height, input_width, output, gamma, beta, mean, variance, epsilon);

    return_value = compare1dFloatArray(output, expected_output, input_height*input_width, error);

    free(output);
    #undef input_width
    #undef input_height



    return return_value;
}

int32_t test_batch_normalization_naive_2() {

    printf("test_batch_normalization_naive_2()\n");
    int32_t return_value = 0;

    #define input_width 3
    #define input_height 3
    fp_t error = 0.001;

    fp_t gamma, beta, mean, variance, epsilon;

    gamma = 1.5;
    beta = 1.1;
    mean = 3;
    variance = 0.9;
    epsilon = 1e-5;

    fp_t input[input_height * input_width] =           {1, 2, 3,
                                                        4, 5, 6,
                                                        7, 8, 9};
    fp_t expected_output[input_height * input_width] = {-2.0622602, -0.48113, 1.1,
                                                        2.68113, 4.26226, 5.84339,
                                                        7.42452, 9.0056505, 10.586781};

    fp_t* output = (fp_t*) malloc(input_height * input_width * sizeof(fp_t));

    batch_normalization_naive(input, input_height, input_width, output, gamma, beta, mean, variance, epsilon);

    return_value = compare1dFloatArray(output, expected_output, input_height * input_width, error);

    free(output);
    #undef input_width
    #undef input_height

    return return_value;
}
