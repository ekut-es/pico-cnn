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
    delete lrn_input_tensor;
    delete lrn_output_tensor;
    delete lrn_expected_output_tensor;
    delete lrn_input_shape;
    delete lrn_output_shape;
    delete lrn_expected_output_shape;
}

void TestActivationFunctions::runTestReLU() {
    fp_t expected_output[10] = {9, 11, 0, 0, 0, 0, 0, 5, 0, 7};
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access(0, i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::ReLU("relu", 0, pico_cnn::op_type::ReLU);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;
}

void TestActivationFunctions::runTestLeakyReLU() {
    PRINT_INFO("Test LeakyReLU...")
    auto lrelu_input_shape = new pico_cnn::naive::TensorShape(1, 11);
    auto lrelu_output_shape = new pico_cnn::naive::TensorShape(1, 11);
    auto lrelu_expected_output_shape = new pico_cnn::naive::TensorShape(1, 11);

    auto lrelu_input_tensor = new pico_cnn::naive::Tensor(lrelu_input_shape);
    auto lrelu_output_tensor = new pico_cnn::naive::Tensor(lrelu_output_shape);
    auto lrelu_expected_output_tensor = new pico_cnn::naive::Tensor(lrelu_expected_output_shape);

    fp_t input[11] = {-7.8, 0.3, -9.9, 6.8, -9.4, 3.6, -9.2, 8.3, 4.3, -1.4, 0.0};
    for(uint32_t i = 0; i < lrelu_input_tensor->num_elements(); i++) {
        lrelu_input_tensor->access_blob(i) = input[i];
    }

    fp_t expected_output[11] = {-0.078, 0.3, -0.099, 6.8, -0.094, 3.6, -0.092, 8.3, 4.3, -0.014, 0.0};
    for(uint32_t i = 0; i < lrelu_expected_output_tensor->num_elements(); i++) {
        lrelu_expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::LeakyReLU("leaky_relu", 0, pico_cnn::op_type::LeakyReLU, 0.01);
    layer->run(lrelu_input_tensor, lrelu_output_tensor);

    CPPUNIT_ASSERT(*lrelu_output_tensor == *lrelu_expected_output_tensor);

    delete layer;

    delete lrelu_input_tensor;
    delete lrelu_output_tensor;
    delete lrelu_expected_output_tensor;
    delete lrelu_input_shape;
    delete lrelu_output_shape;
    delete lrelu_expected_output_shape;
}

void TestActivationFunctions::runTestParameterizedReLU() {
    PRINT_INFO("Test ParamReLU...")
    auto param_relu_input_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto param_relu_output_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto param_relu_expected_output_shape = new pico_cnn::naive::TensorShape(1, 10);

    auto param_relu_input_tensor = new pico_cnn::naive::Tensor(param_relu_input_shape);
    auto param_relu_output_tensor = new pico_cnn::naive::Tensor(param_relu_output_shape);
    auto param_relu_expected_output_tensor = new pico_cnn::naive::Tensor(param_relu_expected_output_shape);

    fp_t input[10] = { -6.4, 1.9, 1.5,  -6.9, -2.0,  -0.3, 8.1,    -2.6, 0.1, -3.8};
    for(uint32_t i = 0; i < param_relu_input_tensor->num_elements(); i++) {
        param_relu_input_tensor->access_blob(i) = input[i];
    }

    fp_t expected_output[10] = {-1.92, 1.9, 1.5, -1.38, -1.2, -0.18, 8.1, -1.8199, 0.1,  0.0};
    for(uint32_t i = 0; i < param_relu_expected_output_tensor->num_elements(); i++) {
        param_relu_expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto slope_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto slope_tensor = new pico_cnn::naive::Tensor(slope_shape);
    fp_t slope[10] = {0.3, 0.9, 0.8, 0.2, 0.6, 0.6, 0.7, 0.7, 0.8, 0.0};
    for(uint32_t i = 0; i < slope_tensor->num_elements(); i++) {
        slope_tensor->access_blob(i) = slope[i];
    }

    auto *layer = new pico_cnn::naive::ParameterizedReLU("param_relu", 0, pico_cnn::op_type::ParamReLU, slope_tensor);
    layer->run(param_relu_input_tensor, param_relu_output_tensor);

    CPPUNIT_ASSERT(*param_relu_output_tensor == *param_relu_expected_output_tensor);

    delete layer;

    delete slope_tensor;
    delete slope_shape;

    delete param_relu_input_tensor;
    delete param_relu_output_tensor;
    delete param_relu_expected_output_tensor;
    delete param_relu_input_shape;
    delete param_relu_output_shape;
    delete param_relu_expected_output_shape;
}

void TestActivationFunctions::runTestSigmoid() {
    PRINT_INFO("Test Sigmoid...")
    auto sigmoid_input_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto sigmoid_output_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto sigmoid_expected_output_shape = new pico_cnn::naive::TensorShape(1, 10);

    auto sigmoid_input_tensor = new pico_cnn::naive::Tensor(sigmoid_input_shape);
    auto sigmoid_output_tensor = new pico_cnn::naive::Tensor(sigmoid_output_shape);
    auto sigmoid_expected_output_tensor = new pico_cnn::naive::Tensor(sigmoid_expected_output_shape);

    fp_t input[10] = {-5.4, 5.8, -4.1, 7.4, 0, 3.4, -4.4, -1.8, 2.2, -0.1};
    for(uint32_t i = 0; i < sigmoid_input_tensor->num_elements(); i++) {
        sigmoid_input_tensor->access_blob(i) = input[i];
    }

    fp_t expected_output[10] = {0.0044962, 0.9969, 0.0163, 0.99939, 0.5, 0.9677, 0.0121, 0.1418, 0.9002, 0.4750};
    for(uint32_t i = 0; i < sigmoid_expected_output_tensor->num_elements(); i++) {
        sigmoid_expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::Sigmoid("sigmoid", 0, pico_cnn::op_type::Sigmoid);
    layer->run(sigmoid_input_tensor, sigmoid_output_tensor);

    CPPUNIT_ASSERT(*sigmoid_output_tensor == *sigmoid_expected_output_tensor);

    delete layer;

    delete sigmoid_input_tensor;
    delete sigmoid_output_tensor;
    delete sigmoid_expected_output_tensor;
    delete sigmoid_input_shape;
    delete sigmoid_output_shape;
    delete sigmoid_expected_output_shape;
}

void TestActivationFunctions::runTestSoftmax() {
    PRINT_INFO("Test Softmax...")
    auto softmax_input_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto softmax_output_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto softmax_expected_output_shape = new pico_cnn::naive::TensorShape(1, 10);

    auto softmax_input_tensor = new pico_cnn::naive::Tensor(softmax_input_shape);
    auto softmax_output_tensor = new pico_cnn::naive::Tensor(softmax_output_shape);
    auto softmax_expected_output_tensor = new pico_cnn::naive::Tensor(softmax_expected_output_shape);

    fp_t input[10] = {0.1, -6.8, -0.4, -0.0, -2.7, 4.5, -5.2, -5.5,  6.9, -0.2};
    for(uint32_t i = 0; i < softmax_input_tensor->num_elements(); i++) {
        softmax_input_tensor->access_blob(i) = input[i];
    }

    fp_t expected_output[10] = {0.001010, 0.000002560, 0.000617, 0.000920, 0.00006188,
                                0.082891, 0.000005079, 0.00000376, 0.913727, 0.0007539};
    for(uint32_t i = 0; i < softmax_expected_output_tensor->num_elements(); i++) {
        softmax_expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::Softmax("softmax", 0, pico_cnn::op_type::Softmax);
    layer->run(softmax_input_tensor, softmax_output_tensor);

    CPPUNIT_ASSERT(*softmax_output_tensor == *softmax_expected_output_tensor);

    delete layer;

    delete softmax_input_tensor;
    delete softmax_output_tensor;
    delete softmax_expected_output_tensor;
    delete softmax_input_shape;
    delete softmax_output_shape;
    delete softmax_expected_output_shape;
}

void TestActivationFunctions::runTestTanH() {
    PRINT_INFO("Test TanH...")
    auto tanh_input_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto tanh_output_shape = new pico_cnn::naive::TensorShape(1, 10);
    auto tanh_expected_output_shape = new pico_cnn::naive::TensorShape(1, 10);

    auto tanh_input_tensor = new pico_cnn::naive::Tensor(tanh_input_shape);
    auto tanh_output_tensor = new pico_cnn::naive::Tensor(tanh_output_shape);
    auto tanh_expected_output_tensor = new pico_cnn::naive::Tensor(tanh_expected_output_shape);

    for(uint32_t i = 0; i < tanh_input_tensor->num_elements(); i++) {
        tanh_input_tensor->access_blob(i) = urand(-1, 1);
        tanh_expected_output_tensor->access_blob(i) = tanhf(tanh_input_tensor->access_blob(i));
    }

    auto *layer = new pico_cnn::naive::TanH("tanh", 0, pico_cnn::op_type::TanH);
    layer->run(tanh_input_tensor, tanh_output_tensor);

    CPPUNIT_ASSERT(*tanh_output_tensor == *tanh_expected_output_tensor);

    delete layer;

    delete tanh_input_tensor;
    delete tanh_output_tensor;
    delete tanh_expected_output_tensor;
    delete tanh_input_shape;
    delete tanh_output_shape;
    delete tanh_expected_output_shape;
}
