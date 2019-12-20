#include <stdio.h>
#include <assert.h>

#include "layers/test_pooling.h"
#include "layers/test_activation_function.h"
#include "layers/test_fully_connected.h"
#include "layers/test_convolution.h"

// int main() {
//
//     printf("Testing: pooling operations...\n");
//     assert(test_max_pooling1d() == 0);
//     assert(test_max_pooling1d_padding() == 0);
//     assert(test_max_pooling2d() == 0);
//     assert(test_max_pooling2d_padding() == 0);
//
//     assert(test_avg_pooling1d() == 0);
//     assert(test_avg_pooling1d_padding() == 0);
//     assert(test_avg_pooling2d() == 0);
//     assert(test_avg_pooling2d_padding() == 0);
//     printf("Success: pooling operations!\n");
//
//     printf("Testing: activation functions...\n");
//     assert(test_relu_naive() == 0);
//     assert(test_leaky_relu_naive() == 0);
//     assert(test_parametrized_relu_naive() == 0);
//     assert(test_sigmoid_naive() == 0);
//     assert(test_softmax_naive() == 0);
//     assert(test_local_response_normalization_naive() == 0);
//     printf("Success: activation functions!\n");
//
//     printf("Testing: fully connected layer...\n");
//     assert(test_fully_connected() == 0);
//     printf("Success: fully connected!\n");
//
//     printf("Testing: convolution operations...\n");
//     assert(test_convolution1d_naive() == 0);
//     assert(test_convolution2d_naive()== 0);
//     printf("Success: convolution!\n");
//
//     printf("All tests failed!\n");
//
//     return 0;
// }

void printSucessFailure(const char* operation, int failed) {
    if(!failed) {
        printf("Success: %s!\n", operation);
    } else {
        printf("Failure: not all tests for %s passed!\n", operation);
    }
}

// assumes that all testing functions return a nonnegative value when they fail
int main() {

    #define separator "------------------------------------------------------------\n"
    int return_value = 0;

    printf(separator);
    printf("Testing: pooling operations...\n");
    int pooling_failed = 0;
    pooling_failed += test_max_pooling1d();
    pooling_failed += test_max_pooling1d_padding();
    pooling_failed += test_max_pooling2d();
    pooling_failed += test_max_pooling2d_padding();

    pooling_failed += test_avg_pooling1d();
    pooling_failed += test_avg_pooling1d_padding();
    pooling_failed += test_avg_pooling2d();
    pooling_failed += test_avg_pooling2d_padding();
    printSucessFailure("pooling operations", pooling_failed);
    return_value += pooling_failed;

    printf(separator);
    printf("Testing: activation functions...\n");
    int activation_functions_failed = 0;
    activation_functions_failed += test_relu_naive();
    activation_functions_failed += test_leaky_relu_naive();
    activation_functions_failed += test_parametrized_relu_naive();
    activation_functions_failed += test_sigmoid_naive();
    activation_functions_failed += test_softmax_naive();
    activation_functions_failed += test_local_response_normalization_naive();
    printSucessFailure("activation functions", activation_functions_failed);
    return_value += activation_functions_failed;

    printf(separator);
    printf("Testing: fully connected layer...\n");
    int fully_connected_failed = 0;
    fully_connected_failed = test_fully_connected();
    printSucessFailure("fully connected layer", fully_connected_failed);
    return_value += fully_connected_failed;

    printf(separator);
    printf("Testing: convolutions...\n");
    int convolution_failed = 0;
    convolution_failed += test_convolution1d_naive();
    convolution_failed += test_convolution2d_naive();
    convolution_failed += test_add_channel2d_naive();
    printSucessFailure("convolutions", convolution_failed);
    return_value += convolution_failed;

    return  return_value;

    #undef separator
}
