#include <stdio.h>
#include <assert.h>

#include "layers/test_pooling.h"
#include "layers/test_activation_function.h"

int main() {

    printf("Testing: pooling operations...\n");
    assert(test_max_pooling1d() == 0);
    assert(test_max_pooling1d_padding() == 0);
    assert(test_max_pooling2d() == 0);
    assert(test_max_pooling2d_padding() == 0);

    assert(test_avg_pooling1d() == 0);
    assert(test_avg_pooling1d_padding() == 0);
    assert(test_avg_pooling2d() == 0);
    assert(test_avg_pooling2d_padding() == 0);
    printf("Success: pooling operations!\n\n");

    assert(test_relu_naive() == 0);
    assert(test_leaky_relu_naive() == 0);
    assert(test_parametrized_relu_naive() == 0);
    assert(test_sigmoid_naive() == 0);
    assert(test_softmax_naive() == 0); 
    printf("Success: activation functions! :) \n\n");

    printf("All tests successful!\n");

    return 0;
}
