#include "test_fully_connected.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestFullyConnected);

void TestFullyConnected::setUp() {
    TestFixture::setUp();

    // It might seem as if the shape should be transposed but onnx uses this layout for gemm kernels as to improve memory access patterns
    input_tensor = new pico_cnn::naive::Tensor(1, 6);
    output_tensor = new pico_cnn::naive::Tensor(1, 4);
    expected_output_tensor = new pico_cnn::naive::Tensor(1, 4);
    kernel_tensor = new pico_cnn::naive::Tensor(4, 6);
    bias_tensor = new pico_cnn::naive::Tensor(4);

    fp_t input[6] = {-2, 4, 1, 8, -5, 0};

    fp_t kernel[24] = {-0.5, -0.1, 0.8, -0.4, -0.3, -0.3,
                       0.2, -1.0, 0.5, -0.9, 0.5, 0.4,
                       -0.3, -0.9, -0.1, -0.9, 0.7, 0.9,
                       0.1, 0.4, -0.4, -0.8, -0.9, -0.7};

    fp_t bias[4] = {6, -3, 0, 1};
    fp_t expected_output[4] = {5.7, -16.6, -13.8, 0.0999};

    for (uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }

    for (uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
        bias_tensor->access_blob(i) = bias[i];
    }

    uint32_t kernel_height = kernel_tensor->height();
    uint32_t kernel_width = kernel_tensor->width();
    for (uint32_t row = 0; row < kernel_height; row++) {
        for (uint32_t col = 0; col < kernel_width; col++) {
            kernel_tensor->access(row, col, kernel_width) = kernel[row * kernel_width + col];
        }
    }

}

void TestFullyConnected::tearDown() {
    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
    delete kernel_tensor;
    delete bias_tensor;

    TestFixture::tearDown();
}

void TestFullyConnected::runTestFullyConnected() {
    //PRINT_INFO("Test FullyConnected...")

    auto *layer = new pico_cnn::naive::FullyConnected("fc", 0, pico_cnn::op_type::Gemm, kernel_tensor, bias_tensor);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;
}
