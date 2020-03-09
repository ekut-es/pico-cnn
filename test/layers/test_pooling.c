#include "test_pooling.h"

void pooling_output_channel_dimensions(uint16_t height, uint16_t width, const uint16_t kernel_size[2],
                                       uint16_t stride, uint16_t **computed_dimensions) {

    uint16_t output_channel_height;
    if(height > 1)
        output_channel_height = (height-kernel_size[0])/stride+1;
    else
        output_channel_height = 1;
    uint16_t output_channel_width = (width-kernel_size[1])/stride+1;

    *computed_dimensions = (uint16_t*) malloc(2 * sizeof(uint16_t));

    (*computed_dimensions)[0] = output_channel_height;
    (*computed_dimensions)[1] = output_channel_width;
}

int32_t test_max_pooling1d() {
    printf("test_max_pooling1d()\n");
    int32_t return_value = 0;

    uint16_t input_height = 1;
    uint16_t input_width = 16;
    uint16_t expected_output_height = 1;
    uint16_t expected_output_width = 7;
    uint16_t kernel_height = 1;
    uint16_t kernel_width = 3;
    uint16_t stride = 2;
    uint16_t kernel_dim[2] = {kernel_height, kernel_width};

    fp_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    fp_t expected_output[7] = {3, 5, 7, 9, 11, 13, 15};

    uint16_t *computed_dimensions;
    pooling_output_channel_dimensions(input_height, input_width, kernel_dim, stride, &computed_dimensions);
    assert(computed_dimensions[0] == expected_output_height);
    assert(computed_dimensions[1] == expected_output_width);
    free(computed_dimensions);

    fp_t* output = (fp_t*) malloc(expected_output_width * sizeof(float));

    max_pooling1d_naive(input, input_width, output, kernel_width, stride);

    for(int32_t i = 0; i < expected_output_width; i++) {
        if(output[i] != expected_output[i]){
            printf("Expected: %f, Output: %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }
    free(output);
    return return_value;
}

int32_t test_max_pooling1d_padding() {
    printf("test_max_pooling1d_padding()\n");
    int32_t return_value = 0;

    uint16_t input_height = 1;
    uint16_t input_width = 16;
    uint16_t expected_output_height = 1;
    uint16_t expected_output_width = 8;
    uint16_t kernel_height = 1;
    uint16_t kernel_width = 3;
    uint16_t stride = 2;
    const uint16_t padding[2] = {1, 1};
    uint16_t kernel_dim[2] = {kernel_height, kernel_width};

    fp_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    fp_t expected_output[8] = {2, 4, 6, 8, 10, 12, 14, 16};

    uint16_t *computed_dimensions;
    pooling_output_channel_dimensions(input_height, input_width+padding[0]+padding[1], kernel_dim, stride, &computed_dimensions);
    assert(computed_dimensions[0] == expected_output_height);
    assert(computed_dimensions[1] == expected_output_width);
    free(computed_dimensions);

    fp_t* output = (fp_t*) malloc(expected_output_width * sizeof(fp_t));


    max_pooling1d_naive_padded(input, input_width, output, kernel_width, stride, padding);

    for(int32_t i = 0; i < expected_output_width; i++) {
        if(expected_output[i] != output[i]){
            printf("Expected: %f, Output; %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }
    free(output);
    return return_value;
}

int32_t test_max_pooling2d() {
    printf("test_max_pooling2d()\n");
    int32_t return_value = 0;

    fp_t input[16] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 15, 16};

    fp_t expected_output[4] = {11, 12,
                               15, 16};

    fp_t* output = (fp_t*) malloc(4* sizeof(fp_t));

    max_pooling2d_naive(input, 4, 4, output, 3, 1);

    for(int32_t i = 0; i < 4; i++) {
        if(output[i] != expected_output[i]) {
            printf("Expected: %f, Output; %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }
    free(output);
    return return_value;
}

int32_t test_max_pooling2d_padding() {
    printf("test_max_pooling2d_padding()\n");
    int32_t return_value = 0;

    fp_t input[25] = {1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25};

    fp_t expected_output[25] = {13, 14, 15, 15, 15,
                                18, 19, 20, 20, 20,
                                23, 24, 25, 25, 25,
                                23, 24, 25, 25, 25,
                                23, 24, 25, 25, 25};

    fp_t* output = (fp_t*) malloc(25* sizeof(float));

    const uint16_t padding[4] = {2, 2, 2, 2};
    max_pooling2d_naive_padded(input, 5, 5, output, 5, 1, padding);

    for(int32_t i = 0; i < 25; i++) {
        if(output[i] != expected_output[i]) {
            printf("Expected: %f, Output; %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }
    free(output);
    return return_value;
}

int32_t test_avg_pooling1d() {
    printf("test_avg_pooling1d()\n");
    int32_t return_value = 0;

    uint16_t input_width = 10;
    uint16_t expected_output_width = 8;
    uint16_t kernel_size = 3;
    uint16_t stride = 1;
    uint16_t count_include_pad = 1;
    uint16_t kernel_dim[2] = {kernel_size, kernel_size};

    uint16_t *computed_dimensions;
    pooling_output_channel_dimensions(1, input_width, kernel_dim, stride, &computed_dimensions);
    assert(computed_dimensions[0] == 1);
    assert(computed_dimensions[1] == expected_output_width);
    free(computed_dimensions);

    fp_t input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    fp_t expected_output[8] = {2, 3, 4, 5, 6, 7, 8, 9};

    fp_t* output = (fp_t*) malloc(1*expected_output_width* sizeof(float));

    average_pooling1d_naive(input, input_width, output, kernel_size, stride,
                            0.0, count_include_pad);

    for(int32_t i = 0; i < 1*expected_output_width; i++){
        if(expected_output[i] != output[i]) {
            printf("Expected: %f, Output; %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }

    free(output);
    return return_value;

}
int32_t test_avg_pooling1d_padding() {
    printf("test_avg_pooling1d_padding()\n");
    int32_t return_value = 0;

    uint16_t input_width = 10;
    uint16_t expected_output_width = 10;
    uint16_t kernel_size = 5;
    uint16_t stride = 1;

    fp_t input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    fp_t expected_output_count_include_pad_0[10] = {2, 2.5, 3, 4, 5, 6, 7, 8, 8.5, 9};

    fp_t expected_output_count_include_pad_1[10] = {1.2, 2, 3, 4, 5, 6, 7, 8, 6.8, 5.4};

    fp_t* output_0 = (fp_t*) malloc(expected_output_width* sizeof(float));
    fp_t* output_1 = (fp_t*) malloc(expected_output_width* sizeof(float));

    const uint16_t padding[4] = {2, 2, 2, 2};

    average_pooling1d_naive_padded(input, input_width, output_0, kernel_size,
                                   stride, 0.0, padding, 0);
    average_pooling1d_naive_padded(input, input_width, output_1, kernel_size,
                                   stride, 0.0, padding, 1);

    for(int32_t i = 0; i < expected_output_width; i++) {
        if (expected_output_count_include_pad_0[i] != output_0[i]) {
            return_value = 1;
        }
        if (expected_output_count_include_pad_1[i] != output_1[i]) {
            return_value = 1;
        }
    }

    free(output_0);
    free(output_1);

    return return_value;
}

int32_t test_avg_pooling2d() {
    printf("test_avg_pooling2d()\n");
    int32_t return_value = 0;

    uint16_t input_height = 5;
    uint16_t input_width = 5;
    uint16_t expected_output_height = 3;
    uint16_t expected_output_width = 3;
    uint16_t kernel_size = 3;
    uint16_t stride = 1;
    uint16_t count_include_pad = 1;
    uint16_t kernel_dim[2] = {kernel_size, kernel_size};

    uint16_t *computed_dimensions;
    pooling_output_channel_dimensions(input_height, input_width, kernel_dim, stride, &computed_dimensions);
    assert(computed_dimensions[0] == expected_output_height);
    assert(computed_dimensions[1] == expected_output_width);
    free(computed_dimensions);


    fp_t input[25] = { 1,  2,  3,  4,  5,
                       6,  7,  8,  9, 10,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25};

    fp_t expected_output[9] = {7, 8, 9,
                               12, 13, 14,
                               17, 18, 19};

    fp_t* output = (fp_t*) malloc(expected_output_height*expected_output_width* sizeof(float));

    average_pooling2d_naive(input, input_height, input_width, output, kernel_size, stride,
                            0.0, count_include_pad);

    for(int32_t i = 0; i < expected_output_height*expected_output_width; i++){
        if(expected_output[i] != output[i]) {
            printf("Expected: %f, Output; %f\n", expected_output[i], output[i]);
            return_value = 1;
        }
    }

    free(output);
    return return_value;
}

int32_t test_avg_pooling2d_padding() {
    printf("test_avg_pooling2d_padding()\n");
    int32_t return_value = 0;

    uint16_t input_height = 5;
    uint16_t input_width = 5;
    //uint16_t expected_output_height = 5;
    //uint16_t expected_output_width = 5;
    uint16_t kernel_size = 5;
    uint16_t stride = 1;

    fp_t input[25] = {1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25};

    fp_t expected_output_count_include_pad_0[25] = {7, 7.5, 8, 8.5, 9,
                                                    9.5, 10, 10.5, 11, 11.5,
                                                    12, 12.5, 13, 13.5, 14,
                                                    14.5, 15, 15.5, 16, 16.5,
                                                    17, 17.5, 18, 18.5, 19};

    fp_t expected_output_count_include_pad_1[25] = {2.5200, 3.6000, 4.8000, 4.0800, 3.2400,
                                                    4.5600, 6.4000, 8.4000, 7.0400, 5.5200,
                                                    7.2000, 10.0000, 13.0000, 10.8000, 8.4000,
                                                    6.9600, 9.6000, 12.4000, 10.2400, 7.9200,
                                                    6.1200, 8.4000, 10.8000, 8.8800, 6.8400};

    fp_t expected_output_2_count_include_pad_0[25] = {4, 4.5, 5.5, 6.5, 7,
                                                      6.5, 7, 8, 9, 9.5,
                                                      11.5, 12, 13, 14, 14.5,
                                                      16.5, 17, 18, 19, 19.5,
                                                      19, 19.5, 20.5, 21.5, 22};

    fp_t* output_0 = (fp_t*) malloc(25* sizeof(float));
    fp_t* output_1 = (fp_t*) malloc(25* sizeof(float));
    fp_t* output_2 = (fp_t*) malloc(25* sizeof(float));

    const uint16_t padding[4] = {2, 2, 2, 2};

    average_pooling2d_naive_padded(input, input_height, input_width, output_0, kernel_size,
                                   stride, 0.0, padding, 0);
    average_pooling2d_naive_padded(input, input_height, input_width, output_1, kernel_size,
                                   stride, 0.0, padding, 1);

    const uint16_t padding2[4] = {1, 1, 1, 1};
    average_pooling2d_naive_padded(input, input_height, input_width, output_2, 3,
                                   stride, 0.0, padding2, 0);

    for(int32_t i = 0; i < 25; i++){
        if(expected_output_count_include_pad_0[i] != output_0[i]) {
            return_value = 1;
            printf("Include pad 0: Index: %d: Expected: %f, Output; %f\n", i, expected_output_count_include_pad_0[i], output_0[i]);
        }

        if(expected_output_count_include_pad_1[i]  != output_1[i]) {
            return_value = 1;
            printf("Include pad 1: Index: %d: Expected: %f, Output; %f\n", i, expected_output_count_include_pad_1[i], output_1[i]);
        }

        if(expected_output_2_count_include_pad_0[i]  != output_2[i]) {
            return_value = 1;
            printf("Include pad 0: Index: %d: Expected: %f, Output; %f\n", i, expected_output_2_count_include_pad_0[i], output_2[i]);
        }
    }

    free(output_0);
    free(output_1);
    free(output_2);
    return return_value;
}

int32_t test_global_average_pooling2d(){

    int32_t return_value = 0;
    printf("test_global_average_pooling2d()\n");

    #define input_width 4
    #define input_height 3
    #define expected_output_size 1
    fp_t error = (fp_t) 0.001;

    fp_t input[input_width * input_height] = {
       -7, 13, -7, -8,
      -15, -4, -7,  4,
        6, -6, -6,  6};

    fp_t expected_output[expected_output_size] = {-2.5833333};

    // just one float
    fp_t* output = malloc(expected_output_size * sizeof(fp_t));

    global_average_pooling2d_naive(input, input_width, input_height, output);

    return_value = compare1dFloatArray(output, expected_output, expected_output_size, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef expected_output_size

}

int32_t test_global_max_pooling2d(){

    int32_t return_value = 0;
    printf("test_global_max_pooling2d()\n");

    #define input_width 5
    #define input_height 4
    #define expected_output_size 1
    fp_t error = (fp_t) 0.001;

    fp_t input[input_width * input_height] = {
       17,  17, -15,  8, -15,
        2,  13,  -2, -5,  6,
        9, -17,  18, 20, -5,
      -14,   4,  11,  7, 18};

    fp_t expected_output[expected_output_size] = {20};

    // just one float
    fp_t* output = malloc(expected_output_size * sizeof(fp_t));

    global_max_pooling2d_naive(input, input_width, input_height, output);

    return_value = compare1dFloatArray(output, expected_output, expected_output_size, error);

    free(output);
    return return_value;

    #undef input_width
    #undef input_height
    #undef expected_output_size

}
