#include "test_activation_functions.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestActivationFunctions);

void TestActivationFunctions::setUp() {
    TestFixture::setUp();

    input_shape = new pico_cnn::naive::TensorShape(1, 10);
    output_shape = new pico_cnn::naive::TensorShape(1, 10);
    expected_output_shape = new pico_cnn::naive::TensorShape(1, 10);

    input_tensor = new pico_cnn::naive::Tensor(input_shape);
    output_tensor = new pico_cnn::naive::Tensor(output_shape);
    expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);

    fp_t input[10] = {9, 11, -4, -5, -9, -4, -7, 5, 0, 7};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access(0, i) = input[i];
    }
}

void TestActivationFunctions::runTestActivationFunction() {
    PRINT_INFO("Test ActivationFunction...")
    fp_t expected_output[10] = {9, 11, -4, -5, -9, -4, -7, 5, 0, 7};
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access(0, i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::ActivationFunction("activate", 0, pico_cnn::op_type::Unknown);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;
}

void TestActivationFunctions::runTestReLU() {
    fp_t expected_output[10] = {9, 11, 0, 0, 0, 0, 0, 5, 0, 7};
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access(0, i) = expected_output[i];
    }

    std::cout << *input_tensor << std::endl;
    std::cout << *expected_output_tensor << std::endl;
    std::cout << *output_tensor << std::endl;

    auto *layer = new pico_cnn::naive::ReLU("relu", 1, pico_cnn::op_type::Relu);
    layer->run(input_tensor, output_tensor);

    std::cout << *output_tensor << std::endl;

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;
}

void TestActivationFunctions::tearDown() {
    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;

    TestFixture::tearDown();
}
