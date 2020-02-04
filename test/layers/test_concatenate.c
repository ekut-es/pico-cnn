#include "test_concatenate.h"

int test_concatenate_2D_1() {

    printf("test_concatenate_2D_1()\n");
    int return_value = 0;

    #define input_width 3
    #define input_height 2
    #define num_inputs 4
    #define input_channel_size 6 // input_width * input_height
    #define output_channel_size 24 // output_channel_size * num_inputs

    fp_t error = 0.0001;
    int dimension = 0;

    fp_t input1[input_channel_size] = { 3,  -9, -16,
                                       14, -18,  19};
    fp_t input2[input_channel_size] = {-9,  10,  -6,
                                        5, -19,  -6};
    fp_t input3[input_channel_size] = { 5, -17,   5,
                                       -5, -13,  -4};
    fp_t input4[input_channel_size] = {12,  10,  15,
                                      -11,  -6,  -6};
    fp_t* input[num_inputs] = {input1, input2, input3, input4};

    fp_t expected_output[output_channel_size] =
        {3, -9, -16,
        14, -18, 19,
        -9, 10, -6,
         5, -19, -6,
         5, -17, 5,
        -5, -13, -4,
        12, 10, 15,
        -11, -6, -6};

    fp_t* output = (fp_t*) malloc(output_channel_size * num_inputs * sizeof(fp_t));

    concatenate_2D(input, input_height, input_width, dimension, num_inputs, output);

    return_value = compare1dFloatArray(output, expected_output, output_channel_size, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef num_inputs
    #undef input_channel_size
    #undef output_channel_size
}

int test_concatenate_2D_2() {

    // same test as in concatenate_2D_1, only dimension is different
    printf("test_concatenate_2D_2()\n");
    int return_value = 0;

    #define input_width 3
    #define input_height 2
    #define num_inputs 4
    #define input_channel_size 6 // input_width * input_height
    #define output_channel_size 24 // input_width * input_channel_size

    fp_t error = 0.0001;
    int dimension = 1;

    fp_t input1[input_channel_size] = { 3,  -9, -16,
                                       14, -18,  19};
    fp_t input2[input_channel_size] = {-9,  10,  -6,
                                        5, -19,  -6};
    fp_t input3[input_channel_size] = { 5, -17,   5,
                                       -5, -13,  -4};
    fp_t input4[input_channel_size] = {12,  10,  15,
                                      -11,  -6,  -6};
    fp_t* input[num_inputs] = {input1, input2, input3, input4};

    fp_t expected_output[output_channel_size] =
        { 3,  -9, -16,-9,  10, -6,  5, -17,  5, 12, 10, 15,
         14, -18,  19, 5, -19, -6, -5, -13, -4,-11, -6, -6};

    fp_t* output = (fp_t*) malloc(output_channel_size * sizeof(fp_t));

    concatenate_2D(input, input_width, input_height, dimension, num_inputs, output);

    return_value = compare1dFloatArray(output, expected_output, output_channel_size, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef num_inputs
    #undef input_channel_size
    #undef output_channel_size
}

int test_concatenate_3D_1() {

    printf("test_concatenate_3D_1()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width 3
    #define input_height 2
    #define num_input_channels 4
    #define num_inputs 2
    #define input_channel_size 6 // (3*2)
    #define num_output_channels 8 // (2*4)

    fp_t error = 0.0001;
    int dimension = 0;

    // input

    fp_t input1[input_channel_size] =   { 3,  -9, -16,
                                         14, -18,  19};
    fp_t input2[input_channel_size] =   {-9,  10,  -6,
                                          5, -19,  -6};
    fp_t input3[input_channel_size] =   { 5, -17,   5,
                                         -5, -13,  -4};
    fp_t input4[input_channel_size] =   {12,  10,  15,
                                        -11,  -6,  -6};
    fp_t input5[input_channel_size] =   { 3,   1,   9,
                                         -4,   5,   9};
    fp_t input6[input_channel_size] =   { 3, -15, -12,
                                          4, -13,   8};
    fp_t input7[input_channel_size] =  {-10, -14,  -1,
                                          2, -10,  10};
    fp_t input8[input_channel_size] =  { -2,  -6, -11,
                                          9,   2,  -6};

    fp_t* input_channel1[num_input_channels] = {input1, input2, input3, input4};
    fp_t* input_channel2[num_input_channels] = {input5, input6, input7, input8};

    fp_t** input[num_inputs] = {input_channel1, input_channel2};

    // expected output

    fp_t expected_output1[input_channel_size] =
       { 3,  -9, -16,
        14, -18,  19};
    fp_t expected_output2[input_channel_size] =
       {-9,  10,  -6,
         5, -19,  -6};
    fp_t expected_output3[input_channel_size] =
       { 5, -17,   5,
        -5, -13,  -4};
    fp_t expected_output4[input_channel_size] =
       {12,  10,  15,
       -11,  -6,  -6};
    fp_t expected_output5[input_channel_size] =
       { 3,   1,   9,
        -4,   5,   9};
    fp_t expected_output6[input_channel_size] =
       { 3, -15, -12,
         4, -13,   8};
    fp_t expected_output7[input_channel_size] =
      {-10, -14,  -1,
         2, -10,  10};
    fp_t expected_output8[input_channel_size] =
      { -2,  -6, -11,
         9,   2,  -6};

    fp_t* expected_output[num_output_channels] =
      {expected_output1,
       expected_output2,
       expected_output3,
       expected_output4,
       expected_output5,
       expected_output6,
       expected_output7,
       expected_output8};

    // allocate memory for output
    fp_t** output = (fp_t**) malloc(num_output_channels*sizeof(fp_t*));

    for(i = 0; i < num_output_channels; i++) {
        output[i] = (fp_t*) malloc(input_channel_size*sizeof(fp_t));
    }

    // run concatenate
    concatenate_3D(input, input_height, input_width, dimension, num_inputs,
                   num_input_channels, output);

    return_value = compare2dFloatArray(output, expected_output, num_output_channels,
                                       input_channel_size, error);

    // free memory for output
    for(i = 0; i < num_output_channels; i++) {
        free(output[i]);
    }

    free(output);

    return return_value;

    #undef input_width
    #undef input_height
    #undef num_input_channels
    #undef num_inputs
    #undef input_channel_size
    #undef num_output_channels
}

int test_concatenate_3D_2() {

    printf("test_concatenate_3D_2()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width 3
    #define input_height 2
    #define num_input_channels 4
    #define num_inputs 2
    #define input_channel_size 6
    #define output_channel_size 12 // input_channel_size * num_inputs


    fp_t error = 0.0001;
    int dimension = 1;

    fp_t input1[input_channel_size] =   { 3,  -9, -16,
                                         14, -18,  19};
    fp_t input2[input_channel_size] =   {-9,  10,  -6,
                                          5, -19,  -6};
    fp_t input3[input_channel_size] =   { 5, -17,   5,
                                         -5, -13,  -4};
    fp_t input4[input_channel_size] =   {12,  10,  15,
                                        -11,  -6,  -6};
    fp_t input5[input_channel_size] =   { 3,   1,   9,
                                         -4,   5,   9};
    fp_t input6[input_channel_size] =   { 3, -15, -12,
                                          4, -13,   8};
    fp_t input7[input_channel_size] =  {-10, -14,  -1,
                                          2, -10,  10};
    fp_t input8[input_channel_size] =  { -2,  -6, -11,
                                          9,   2,  -6};

    fp_t* input_channel1[num_input_channels] = {input1, input2, input3, input4};
    fp_t* input_channel2[num_input_channels] = {input5, input6, input7, input8};

    fp_t** input[num_inputs] = {input_channel1, input_channel2};

    fp_t expected_output1[output_channel_size ] =  {3,  -9, -16,
                                                   14, -18,  19,
                                                    3,   1,   9,
                                                   -4,   5,   9};
    fp_t expected_output2[output_channel_size ] = {-9,  10,  -6,
                                                    5, -19,  -6,
                                                    3, -15, -12,
                                                    4, -13,   8};
    fp_t expected_output3[output_channel_size ] =  {5, -17,   5,
                                                   -5, -13,  -4,
                                                  -10, -14,  -1,
                                                    2, -10,  10};
    fp_t expected_output4[output_channel_size ] = {12,  10,  15,
                                                  -11,  -6,  -6,
                                                   -2,  -6, -11,
                                                    9,   2,  -6};

    fp_t* expected_output[num_input_channels] =
      {expected_output1,expected_output2,expected_output3, expected_output4};

    // allocate memory for output
    fp_t** output = (fp_t**) malloc(num_input_channels*sizeof(fp_t*));

    for(i = 0; i < num_input_channels; i++) {
        output[i] = (fp_t*) malloc(output_channel_size*sizeof(fp_t));
    }

    concatenate_3D(input, input_height, input_width, dimension, num_inputs,
                   num_input_channels, output);

    return_value = compare2dFloatArray(output, expected_output,input_height,
                                       input_width, error);

    // free memory
    for(i = 0; i < num_input_channels; i++) {
        free(output[i]);
    }
    free(output);

    return return_value;

    #undef input_width
    #undef input_height
    #undef num_input_channels
    #undef num_inputs
    #undef input_channel_size
    #undef output_channel_size
}

int test_concatenate_3D_3() {

    printf("test_concatenate_3D_3()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width 3
    #define input_height 2
    #define num_input_channels 4
    #define num_inputs 2
    #define input_channel_size 6 // input_width * input_height
    #define output_channel_size 12 // input_channel_size * num_inputs
    #define num_output_channels 4

    fp_t error = 0.0001;
    int dimension = 2;

    // input

    fp_t input1[input_channel_size] =   { 3,  -9, -16,
                                         14, -18,  19};
    fp_t input2[input_channel_size] =   {-9,  10,  -6,
                                          5, -19,  -6};
    fp_t input3[input_channel_size] =   { 5, -17,   5,
                                         -5, -13,  -4};
    fp_t input4[input_channel_size] =   {12,  10,  15,
                                        -11,  -6,  -6};
    fp_t input5[input_channel_size] =   { 3,   1,   9,
                                         -4,   5,   9};
    fp_t input6[input_channel_size] =   { 3, -15, -12,
                                          4, -13,   8};
    fp_t input7[input_channel_size] =  {-10, -14,  -1,
                                          2, -10,  10};
    fp_t input8[input_channel_size] =  { -2,  -6, -11,
                                          9,   2,  -6};

    fp_t* input_channel1[num_input_channels] = {input1, input2, input3, input4};
    fp_t* input_channel2[num_input_channels] = {input5, input6, input7, input8};

    fp_t** input[num_inputs] = {input_channel1, input_channel2};

    // expected output
    fp_t expected_output1[output_channel_size] =  {3, -9, -16 , 3,   1,   9,
                                                 14, -18,  19, -4,   5,   9};
    fp_t expected_output2[output_channel_size] = {-9, 10,  -6,  3, -15, -12,
                                                  5, -19,  -6,  4, -13,   8};
    fp_t expected_output3[output_channel_size] = {5, -17,   5,-10, -14,  -1,
                                                 -5, -13,  -4,  2, -10,  10};
    fp_t expected_output4[output_channel_size] = {12, 10,  15, -2,  -6, -11,
                                                 -11, -6,  -6,  9,   2,  -6};

    fp_t* expected_output[num_output_channels] =
      {expected_output1,expected_output2,expected_output3, expected_output4};

    // allocate memory for output
    fp_t** output = (fp_t**) malloc(num_input_channels * sizeof(fp_t*));
    for(i = 0; i < num_input_channels; i++) {
        output[i] = (fp_t*) malloc(output_channel_size * sizeof(fp_t));
    }

    concatenate_3D(input, input_height, input_width, dimension, num_inputs,
                   num_input_channels, output);

//    print2dFloatArray_3d(output, num_input_channels, input_height, input_width);

    return_value = compare2dFloatArray(output, expected_output,num_input_channels,
                                       output_channel_size, error);

    // free memory
    for(i = 0; i < num_input_channels; i++) {
        free(output[i]);
    }
    free(output);

    return return_value;

    #undef input_width
    #undef input_height
    #undef num_input_channels
    #undef num_inputs
    #undef input_channel_size
    #undef output_channel_size
    #undef num_output_channels
}

int test_concatenate_naive_dim_0() {

    printf("test_concatenate_naive_dim_0()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width 3
    #define input_height 3
    #define num_input_channels_0 2
    #define num_input_channels_1 3
    #define num_inputs 2
    #define input_channel_size 9 // (3*3)
    #define num_output_channels 5 // (2+3)

    fp_t error = 0.0001;
    int dimension = 0;

    // input

    fp_t input_0_0[input_channel_size] =   {1, 2, 3,
                                            4, 5, 6,
                                            7, 8, 9};
    fp_t input_0_1[input_channel_size] =   {10, 11, 12,
                                            13, 14, 15,
                                            16, 17, 18};

    fp_t input_1_0[input_channel_size] =   {19, 20, 21,
                                            22, 23, 24,
                                            25, 26, 27};
    fp_t input_1_1[input_channel_size] =   {28, 29, 30,
                                            31, 32, 33,
                                            34, 35, 36};
    fp_t input_1_2[input_channel_size] =   {37, 38, 39,
                                            40, 41, 42,
                                            43, 44, 45};


    fp_t* input_channel_0[num_input_channels_0] = {input_0_0, input_0_1};
    fp_t* input_channel_1[num_input_channels_1] = {input_1_0, input_1_1, input_1_2};

    fp_t** input[num_inputs] = {input_channel_0, input_channel_1};

    // expected output
    fp_t expected_output1[input_channel_size] =
            {1, 2, 3,
             4, 5, 6,
             7, 8, 9};
    fp_t expected_output2[input_channel_size] =
            {10, 11, 12,
             13, 14, 15,
             16, 17, 18};
    fp_t expected_output3[input_channel_size] =
            {19, 20, 21,
             22, 23, 24,
             25, 26, 27};
    fp_t expected_output4[input_channel_size] =
            {28, 29, 30,
             31, 32, 33,
             34, 35, 36};
    fp_t expected_output5[input_channel_size] =
            {37, 38, 39,
             40, 41, 42,
             43, 44, 45};

    fp_t* expected_output[num_output_channels] =
            {expected_output1,
             expected_output2,
             expected_output3,
             expected_output4,
             expected_output5};

    // allocate memory for output
    fp_t** output = (fp_t**) malloc(num_output_channels*sizeof(fp_t*));

    for(i = 0; i < num_output_channels; i++) {
        output[i] = (fp_t*) malloc(input_channel_size*sizeof(fp_t));
    }

    uint16_t input_shape_0[3] = {2, 3, 3};
    uint16_t input_shape_1[3] = {3, 3, 3};
    const uint16_t* input_shape[num_inputs] = {input_shape_0, input_shape_1};

    // run concatenate
    concatenate_naive(input, input_shape, dimension, num_inputs, output);

//    print2dFloatArray_3d(output, num_output_channels, input_height, input_width);

    return_value = compare2dFloatArray(output, expected_output, num_output_channels,
                                       input_channel_size, error);

    // free memory for output
    for(i = 0; i < num_output_channels; i++) {
        free(output[i]);
    }
    free(output);

    return return_value;

    #undef input_width
    #undef input_height
    #undef num_input_channels_0
    #undef num_input_channels_1
    #undef num_inputs
    #undef input_channel_size
    #undef num_output_channels
}

int test_concatenate_naive_dim_1() {

    printf("test_concatenate_naive_dim_1()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width_0 3
    #define input_height_0 3
    #define input_width_1 3
    #define input_height_1 4
    #define num_input_channels_0 2
    #define num_input_channels_1 2
    #define num_inputs 2
    #define input_channel_size_0 9 // (3*3)
    #define input_channel_size_1 12 // (4*3)
    #define output_channel_size 21
    #define num_output_channels 2

    fp_t error = 0.0001;
    int dimension = 1;

    // input

    fp_t input_0_0[input_channel_size_0] =   {1, 2, 3,
                                              4, 5, 6,
                                              7, 8, 9};
    fp_t input_0_1[input_channel_size_0] =   {10, 11, 12,
                                              13, 14, 15,
                                              16, 17, 18};

    fp_t input_1_0[input_channel_size_1] =   {19, 20, 21,
                                              22, 23, 24,
                                              25, 26, 27,
                                              0, 0, 0};
    fp_t input_1_1[input_channel_size_1] =   {28, 29, 30,
                                              31, 32, 33,
                                              34, 35, 36,
                                              -1, -1, -1};


    fp_t* input_channel_0[num_input_channels_0] = {input_0_0, input_0_1};
    fp_t* input_channel_1[num_input_channels_1] = {input_1_0, input_1_1};

    fp_t** input[num_inputs] = {input_channel_0, input_channel_1};

    // expected output
    fp_t expected_output1[output_channel_size] =
            {1, 2, 3,
             4, 5, 6,
             7, 8, 9,
             19, 20, 21,
             22, 23, 24,
             25, 26, 27,
             0, 0, 0};
    fp_t expected_output2[output_channel_size] =
            {10, 11, 12,
             13, 14, 15,
             16, 17, 18,
             28, 29, 30,
             31, 32, 33,
             34, 35, 36,
             -1, -1, -1};

    fp_t* expected_output[num_output_channels] =
            {expected_output1,
             expected_output2};

    // allocate memory for output
    fp_t** output = (fp_t**) malloc(num_output_channels*sizeof(fp_t*));

    for(i = 0; i < num_output_channels; i++) {
        output[i] = (fp_t*) malloc(output_channel_size*sizeof(fp_t));
    }

    uint16_t input_shape_0[3] = {2, 3, 3};
    uint16_t input_shape_1[3] = {2, 4, 3};
    const uint16_t* input_shape[num_inputs] = {input_shape_0, input_shape_1};

    // run concatenate
    concatenate_naive(input, input_shape, dimension, num_inputs, output);

//    print2dFloatArray_3d(output, num_output_channels, 7, 3);

    return_value = compare2dFloatArray(output, expected_output, num_output_channels,
                                       output_channel_size, error);

    // free memory for output
    for(i = 0; i < num_output_channels; i++) {
        free(output[i]);
    }
    free(output);

    return return_value;

    #undef input_width_0
    #undef input_height_0
    #undef input_width_1
    #undef input_height_1
    #undef num_input_channels_0
    #undef num_input_channels_1
    #undef num_inputs
    #undef input_channel_size_0
    #undef input_channel_size_1
    #undef num_output_channels
}

int test_concatenate_naive_dim_2(){

    printf("test_concatenate_naive_dim_2()\n");
    int return_value = 0;
    uint16_t i;

    #define input_width_0 3
    #define input_height_0 4
    #define input_width_1 3
    #define input_height_1 3
    #define num_input_channels_0 2
    #define num_input_channels_1 2
    #define num_inputs 2
    #define input_channel_size_0 12 // (3*4)
    #define input_channel_size_1 9 // (3*3)
    #define output_channel_size 21 // 3*(3+4)
    #define num_output_channels 2

    fp_t error = 0.0001;
    int dimension = 2;

    // input

    fp_t input_0_0[input_channel_size_0] =   {1, 2, 3, 0,
                                              4, 5, 6, 0,
                                              7, 8, 9, 0};
    fp_t input_0_1[input_channel_size_0] =   {10, 11, 12, -1,
                                              13, 14, 15, -1,
                                              16, 17, 18, -1};

    fp_t input_1_0[input_channel_size_1] =   {19, 20, 21,
                                              22, 23, 24,
                                              25, 26, 27};
    fp_t input_1_1[input_channel_size_1] =   {28, 29, 30,
                                              31, 32, 33,
                                              34, 35, 36};


    fp_t* input_channel_0[num_input_channels_0] = {input_0_0, input_0_1};
    fp_t* input_channel_1[num_input_channels_1] = {input_1_0, input_1_1};

    fp_t** input[num_inputs] = {input_channel_0, input_channel_1};

    // expected output
    fp_t expected_output1[output_channel_size] = {1, 2, 3, 0, 19, 20, 21,
                                                  4, 5, 6, 0, 22, 23, 24,
                                                  7, 8, 9, 0, 25, 26, 27};
    fp_t expected_output2[output_channel_size] = {10, 11, 12, -1, 28, 29, 30,
                                                  13, 14, 15, -1, 31, 32, 33,
                                                  16, 17, 18, -1, 34, 35, 36};

    fp_t* expected_output[num_output_channels] = {expected_output1,
                                                  expected_output2};

    // allocate memory for output
    fp_t** output = (fp_t**) malloc(num_output_channels*sizeof(fp_t*));

    for(i = 0; i < num_output_channels; i++) {
        output[i] = (fp_t*) malloc(output_channel_size*sizeof(fp_t));
    }

    uint16_t input_shape_0[3] = {2, 3, 4};
    uint16_t input_shape_1[3] = {2, 3, 3};
    const uint16_t* input_shape[num_inputs] = {input_shape_0, input_shape_1};

    // run concatenate
    concatenate_naive(input, input_shape, dimension, num_inputs, output);

//    print2dFloatArray_3d(output, num_output_channels, 3, 7);

    return_value = compare2dFloatArray(output, expected_output, num_output_channels,
                                       output_channel_size, error);

    // free memory for output
    for(i = 0; i < num_output_channels; i++) {
        free(output[i]);
    }
    free(output);

    return return_value;

    #undef input_width_0
    #undef input_height_0
    #undef input_width_1
    #undef input_height_1
    #undef num_input_channels_0
    #undef num_input_channels_1
    #undef num_inputs
    #undef input_channel_size_0
    #undef input_channel_size_1
    #undef num_output_channels
}