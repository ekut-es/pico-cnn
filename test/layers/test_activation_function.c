#include "test_activation_function.h"

int test_relu_naive() {

  printf("test_relu_naive()\n");
  int return_value = 0;

  #define input_width 10
  #define input_height 1
  #define expected_output_height 1
  #define expected_output_width 10

  fp_t input[input_width] = {9, 10, -4, -5, -9, -4, -7, 5, -7, 7};
  fp_t expected_output[expected_output_width] = {9,10,0,0,0,0,0,5,0,7};

  assert(input_width == expected_output_width);

  fp_t* output = malloc(expected_output_width * sizeof(float));

  relu_naive(input, input_height, input_width, output);

  for(int i = 0; i < expected_output_width; i++) {
    if(output[i] != expected_output[i]) {
        printf("Expected: %f, Output: %f\n", expected_output[i], output[i]);
        return_value = 1;
    }
  }
  free(output);
  return return_value;
}
