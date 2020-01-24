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

int test_concatenate_3D_1() {

    printf("test_concatenate_3D_1()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width 3
    #define input_height 2
    #define num_input_channels 4
    #define num_inputs 2

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
    fp_t input5[input_width*input_height] = {3, 1, 9,
                                             -4, 5, 9};
    fp_t input6[input_width*input_height] = {3, -15, -12,
                                             4, -13, 8};
    fp_t input7[input_width*input_height] = {-10, -14, -1,
                                              2, -10, 10};
    fp_t input8[input_width*input_height] = {-2, -6, -11,
                                              9, 2, -6};

    fp_t* input_channel1[num_input_channels] = {input1, input2, input3, input4};
    fp_t* input_channel2[num_input_channels] = {input5, input6, input7, input8};

    fp_t** input[num_inputs] = {input_channel1, input_channel2};

    fp_t expected_output1[input_width*input_height] = {3, -9, -16,
                                             14, -18, 19};
    fp_t expected_output2[input_width*input_height] = {-9, 10, -6,
                                              5, -19, -6};
    fp_t expected_output3[input_width*input_height] = {5, -17, 5,
                                            -5, -13, -4};
    fp_t expected_output4[input_width*input_height] = {12, 10, 15,
                                            -11, -6, -6};
    fp_t expected_output5[input_width*input_height] = {3, 1, 9,
                                             -4, 5, 9};
    fp_t expected_output6[input_width*input_height] = {3, -15, -12,
                                             4, -13, 8};
    fp_t expected_output7[input_width*input_height] = {-10, -14, -1,
                                              2, -10, 10};
    fp_t expected_output8[input_width*input_height] = {-2, -6, -11,
                                              9, 2, -6};
    fp_t* expected_output[num_inputs * num_input_channels] =
      {expected_output1,expected_output2,expected_output3, expected_output4,
       expected_output5, expected_output6, expected_output7,expected_output8};

    fp_t** output = (fp_t**) malloc(num_inputs*num_input_channels*sizeof(fp_t*));
    for(i = 0; i < num_inputs * num_input_channels; i++) {
        output[i] = (fp_t*) malloc(input_width *input_height*sizeof(fp_t));
    }


    concatenate_3D(input, input_width, input_height, dimension, num_inputs,
                   num_input_channels, output);

    print2dFloatArray_3d(output, num_input_channels, input_height, input_width);

    return_value = compare2dFloatArray(output, expected_output,input_height,
                                       input_width, error);

    for(i = 0; i < num_inputs*num_input_channels; i++) {
        free(output[i]);
    }
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef num_input_channels
    #undef num_inputs
}

int test_concatenate_3D_2() {

    printf("test_concatenate_3D_2()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width 3
    #define input_height 2
    #define num_input_channels 4
    #define num_inputs 2

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
    fp_t input5[input_width*input_height] = {3, 1, 9,
                                             -4, 5, 9};
    fp_t input6[input_width*input_height] = {3, -15, -12,
                                             4, -13, 8};
    fp_t input7[input_width*input_height] = {-10, -14, -1,
                                              2, -10, 10};
    fp_t input8[input_width*input_height] = {-2, -6, -11,
                                              9, 2, -6};

    fp_t* input_channel1[num_input_channels] = {input1, input2, input3, input4};
    fp_t* input_channel2[num_input_channels] = {input5, input6, input7, input8};

    fp_t** input[num_inputs] = {input_channel1, input_channel2};

    fp_t expected_output1[input_width*input_height*num_inputs] = {3, -9, -16,
                                             14, -18, 19,
                                             3, 1, 9,
                                            -4, 5, 9};
    fp_t expected_output2[input_width*input_height*num_inputs] = {-9, 10, -6,
                                              5, -19, -6,
                                              3, -15, -12,
                                             4, -13, 8};
    fp_t expected_output3[input_width*input_height*num_inputs] = {5, -17, 5,
                                            -5, -13, -4,
                                            -10, -14, -1,
                                            2, -10, 10};
    fp_t expected_output4[input_width*input_height*num_inputs] = {12, 10, 15,
                                            -11, -6, -6,
                                            -2, -6, -11,
                                              9, 2, -6};

    fp_t* expected_output[num_input_channels] =
      {expected_output1,expected_output2,expected_output3, expected_output4};

    fp_t** output = (fp_t**) malloc(num_input_channels*sizeof(fp_t*));
    for(i = 0; i < num_input_channels; i++) {
        output[i] = (fp_t*) malloc(input_width *input_height*num_inputs*sizeof(fp_t));
    }


    concatenate_3D(input, input_width, input_height, dimension, num_inputs,
                   num_input_channels, output);

    print2dFloatArray_3d(output, num_input_channels, input_height * num_inputs, input_width);

    return_value = compare2dFloatArray(output, expected_output,input_height,
                                       input_width, error);

    for(i = 0; i < num_input_channels; i++) {
        free(output[i]);
    }
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef num_input_channels
    #undef num_inputs
}

int test_concatenate_3D_3() {

    printf("test_concatenate_3D_3()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width 3
    #define input_height 2
    #define num_input_channels 4
    #define num_inputs 2

    fp_t error = 0.0001;
    int dimension = 2;

    fp_t input1[input_width*input_height] = {3, -9, -16,
                                             14, -18, 19};
    fp_t input2[input_width*input_height] = {-9, 10, -6,
                                              5, -19, -6};
    fp_t input3[input_width*input_height] = {5, -17, 5,
                                            -5, -13, -4};
    fp_t input4[input_width*input_height] = {12, 10, 15,
                                            -11, -6, -6};
    fp_t input5[input_width*input_height] = {3, 1, 9,
                                             -4, 5, 9};
    fp_t input6[input_width*input_height] = {3, -15, -12,
                                             4, -13, 8};
    fp_t input7[input_width*input_height] = {-10, -14, -1,
                                              2, -10, 10};
    fp_t input8[input_width*input_height] = {-2, -6, -11,
                                              9, 2, -6};

    fp_t* input_channel1[num_input_channels] = {input1, input2, input3, input4};
    fp_t* input_channel2[num_input_channels] = {input5, input6, input7, input8};

    fp_t** input[num_inputs] = {input_channel1, input_channel2};

    fp_t expected_output1[input_width*input_height*num_inputs] = {3, -9, -16 ,3, 1, 9,
                                                                 14, -18, 19, -4, 5, 9};
    fp_t expected_output2[input_width*input_height*num_inputs] = {-9, 10, -6, 3, -15, -12,
                                                                  5, -19, -6, 4, -13, 8};
    fp_t expected_output3[input_width*input_height*num_inputs] = {5, -17, 5,-10, -14, -1,
                                                                 -5, -13, -4, 2, -10, 10};
    fp_t expected_output4[input_width*input_height*num_inputs] = {12, 10, 15,-2, -6, -11,
                                                                  -11, -6, -6,9, 2, -6};

    fp_t* expected_output[num_inputs * num_input_channels] =
      {expected_output1,expected_output2,expected_output3, expected_output4};

    fp_t** output = (fp_t**) malloc(num_input_channels*sizeof(fp_t*));
    for(i = 0; i < num_input_channels; i++) {
        output[i] = (fp_t*) malloc(input_width *input_height*num_inputs*sizeof(fp_t));
    }

    concatenate_3D(input, input_width, input_height, dimension, num_inputs,
                   num_input_channels, output);

    print2dFloatArray_3d(output, num_input_channels, input_height, input_width);

    return_value = compare2dFloatArray(output, expected_output,input_height,
                                       input_width, error);

    for(i = 0; i < num_input_channels; i++) {
        free(output[i]);
    }
    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef num_input_channels
    #undef num_inputs
}
