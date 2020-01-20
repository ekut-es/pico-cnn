#include "test_concatenate.h"

int test_concatenate_2D_1() {

    printf("test_concatenate_2D_1()\n");
    int return_value = 0;

    #define input_width 3
    #define input_height 2
    #define num_inputs 4

    fp_t error = 0.0001;
    int dimension = 0;

    fp_t input1[input_width*input_height] = {3, -9, -16,
                                             14, -18, 19};
    fp_t input2[input_width*input_height] = {-9, 10, -6,
                                              5, -19, -6};
    fp_t input3[input_width*input_height] = {5, -17, 5,
                                            -5, -13, -4};
    fp_t input4[input_width*input_height] = {12, 10, 15,
                                            -11, -6, -6};
    fp_t* input[num_inputs] = {input1, input2, input3, input4};

    fp_t expected_output[input_width*input_height*num_inputs] =
        {3, -9, -16,
        14, -18, 19,
        -9, 10, -6,
         5, -19, -6,
         5, -17, 5,
        -5, -13, -4,
        12, 10, 15,
        -11, -6, -6};

    fp_t* output = (fp_t*) malloc(input_width * input_height * num_inputs * sizeof(fp_t));

    concatenate_2D(input, input_width, input_height, dimension, num_inputs, output);

    return_value = compare1dFloatArray(output, expected_output, input_width*input_height*num_inputs, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef num_inputs
}

int test_concatenate_2D_2() {

    // same test as in concatenate_2D_1, only dimension is different
    printf("test_concatenate_2D_2()\n");
    int return_value = 0;

    #define input_width 3
    #define input_height 2
    #define num_inputs 4

    fp_t error = 0.0001;
    int dimension = 1;

    fp_t input1[input_width*input_height] = {3, -9, -16,
                                             14, -18, 19};
    fp_t input2[input_width*input_height] = {-9, 10, -6,
                                              5, -19, -6};
    fp_t input3[input_width*input_height] = {5, -17, 5,
                                            -5, -13, -4};
    fp_t input4[input_width*input_height] = {12, 10, 15,
                                            -11, -6, -6};
    fp_t* input[num_inputs] = {input1, input2, input3, input4};

    fp_t expected_output[input_width*input_height*num_inputs] =
        {3, -9, -16,-9, 10, -6, 5, -17, 5,12, 10, 15,
         14, -18, 19, 5, -19, -6,-5, -13, -4,-11, -6, -6};

    fp_t* output = (fp_t*) malloc(input_width * input_height * num_inputs * sizeof(fp_t));

    concatenate_2D(input, input_width, input_height, dimension, num_inputs, output);

    return_value = compare1dFloatArray(output, expected_output, input_width*input_height*num_inputs, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef num_inputs
}
