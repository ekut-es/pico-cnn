#include <stdio.h>
#include <assert.h>

#include "layers/test_pooling.h"

int main() {

    printf("Testing: pooling operations...\n");
    assert(test_max_pooling1d() == 0);
    assert(test_max_pooling1d_padding() == 0);
    assert(test_max_pooling2d() == 0);
    assert(test_max_pooling2d_padding() == 0);
    assert(test_avg_pooling2d() == 0);
    assert(test_avg_pooling2d_padding() == 0);
    printf("Success: pooling operations!\n");

    printf("All tests successful!\n");

    return 0;
}
