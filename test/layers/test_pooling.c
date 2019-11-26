#include "test_pooling.h"

int test_max_pooling2d() {
    int return_value = 0;

    fp_t input[16] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 15, 16};

    fp_t expected_output[4] = {11, 12,
                               15, 16};

    fp_t* output = (fp_t*) malloc(4* sizeof(float));

    max_pooling2d_naive(input, 4, 4, output, 3, 1);

    for(int i = 0; i < 4; i++) {
        if(output[i] != expected_output[i]) {
            return_value = 1;
        }
    }
    free(output);
    return return_value;
}

int test_max_pooling2d_padding() {
    int return_value = 0;

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

    const int padding[4] = {2, 2, 2, 2};
    max_pooling2d_naive_padded(input, 5, 5, output, 5, 1, padding);

    for(int i = 0; i < 25; i++) {
        if(output[i] != expected_output[i]) {
            return_value = 1;
        }
    }
    free(output);
    return return_value;
}
