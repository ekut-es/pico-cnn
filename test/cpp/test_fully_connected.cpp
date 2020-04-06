#include "test_fully_connected.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestFullyConnected);

void TestFullyConnected::setUp() {
    TestFixture::setUp();

    input_shape = new pico_cnn::naive::TensorShape(1, 6);
    output_shape = new pico_cnn::naive::TensorShape(1, 4);
    expected_output_shape = new pico_cnn::naive::TensorShape(1, 4);
    kernel_shape = new pico_cnn::naive::TensorShape(6, 4);
    bias_shape = new pico_cnn::naive::TensorShape(1, 4);

    input_tensor = new pico_cnn::naive::Tensor(input_shape);
    output_tensor = new pico_cnn::naive::Tensor(output_shape);
    expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);
    bias_tensor = new pico_cnn::naive::Tensor(bias_shape);

    fp_t input[6] = {-2, 4, 1, 8, -5, 0};
    fp_t kernel[24] = {-0.5,  0.2, -0.3,  0.1,
                       -0.1, -1.0, -0.9,  0.4,
                        0.8,  0.5, -0.1, -0.4,
                       -0.4, -0.9, -0.9, -0.8,
                       -0.3,  0.5,  0.7, -0.9,
                       -0.3,  0.4,  0.9, -0.7};
    fp_t bias[4] = {6, -3, 0, 1};
    fp_t expected_output[4] = {5.6999, -16.6, -13.8, 0.0999};

    for (uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }

    for (uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
        bias_tensor->access_blob(i) = bias[i];
    }

    uint32_t kernel_height = kernel_tensor->shape()->operator[](0);
    uint32_t kernel_width = kernel_tensor->shape()->operator[](1);
    for (uint32_t row = 0; row < kernel_height; row++) {
        for (uint32_t col = 0; col < kernel_width; col++) {
            kernel_tensor->access(row, col) = kernel[row * kernel_width + col];
        }
    }

}

void TestFullyConnected::tearDown() {
    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
    delete kernel_tensor;
    delete bias_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;
    delete kernel_shape;
    delete bias_shape;

    TestFixture::tearDown();
}

void TestFullyConnected::runTestFullyConnected() {
    PRINT_INFO("Test FullyConnected...")

    auto *layer = new pico_cnn::naive::FullyConnected("fc", 0, pico_cnn::op_type::Gemm, kernel_tensor, bias_tensor);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;
}
