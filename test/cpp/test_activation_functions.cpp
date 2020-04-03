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

void TestActivationFunctions::tearDown() {
    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;

    TestFixture::tearDown();
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

void TestActivationFunctions::runTestClip() {
    PRINT_INFO("Test Clip...")
    fp_t expected_output[10] = {9, 9, -4, -5, -7, -4, -7, 5, 0, 7};
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access(0, i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::Clip("clip", 0, pico_cnn::op_type::Clip, -7, 9);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;
}

void TestActivationFunctions::runTestLRN() {
    PRINT_INFO("Test LRN...")
    auto lrn_input_shape = new pico_cnn::naive::TensorShape(3,2, 3);
    auto lrn_output_shape = new pico_cnn::naive::TensorShape(3,2, 3);
    auto lrn_expected_output_shape = new pico_cnn::naive::TensorShape(3,2, 3);

    auto lrn_input_tensor = new pico_cnn::naive::Tensor(lrn_input_shape);
    auto lrn_output_tensor = new pico_cnn::naive::Tensor(lrn_output_shape);
    auto lrn_expected_output_tensor = new pico_cnn::naive::Tensor(lrn_expected_output_shape);

    fp_t input[18] = {-1, -2,  2,
                      0, -2, -2,

                      -1, -5, -1,
                      2,  1,  0,

                      -1,  1, -3,
                      2,  5, -4};
    for(uint32_t i = 0; i < lrn_input_tensor->num_elements(); i++) {
        lrn_input_tensor->access_blob(i) = input[i];
    }

    fp_t expected_output[18] = {-0.95342, -1.2777, 1.7888,
                                0, -1.7888,-1.8257,

                                -0.9325, -3.1622,-0.76694,
                                1.69030 ,0.63245,      0,

                                -0.9534625,0.659380, -2.44948,
                                1.69030, 3.29690,-2.981423};
    for(uint32_t i = 0; i < lrn_expected_output_tensor->num_elements(); i++) {
        lrn_expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::LRN("lrn", 0, pico_cnn::op_type::LRN, 0.1, 0.5, 2);
    layer->run(lrn_input_tensor, lrn_output_tensor);

    CPPUNIT_ASSERT(*lrn_output_tensor == *lrn_expected_output_tensor);

    delete layer;
}

void TestActivationFunctions::runTestReLU() {
    fp_t expected_output[10] = {9, 11, 0, 0, 0, 0, 0, 5, 0, 7};
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access(0, i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::ReLU("relu", 0, pico_cnn::op_type::Relu);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;
}

void TestActivationFunctions::runTestLeakyReLU() {

}

void TestActivationFunctions::runTestParameterizedReLU() {

}

void TestActivationFunctions::runTestSigmoid() {

}

void TestActivationFunctions::runTestSoftmax() {

}

void TestActivationFunctions::runTestTanH() {

}


