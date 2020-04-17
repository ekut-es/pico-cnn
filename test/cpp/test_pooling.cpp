#include "test_pooling.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestPooling);

void TestPooling::setUp() {
    TestFixture::setUp();
}

void TestPooling::tearDown() {
    TestFixture::tearDown();
}

void TestPooling::runTestMaxPooling2d() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 4, 4);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 2, 2);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 2, 2);

    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);

    fp_t input[16] = {1, 2, 3, 4,
                      5, 6, 7, 8,
                      9, 10, 11, 12,
                      13, 14, 15, 16};

    fp_t expected_output[4] = {11, 12,
                               15, 16};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::MaxPooling("MaxPool", 0, pico_cnn::op_type::MaxPool,3, 1, nullptr);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;
}

void TestPooling::runTestMaxPooling2dPadding() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);

    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);

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

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    uint32_t padding[4] = {2, 2, 2, 2};

    auto *layer = new pico_cnn::naive::MaxPooling("MaxPool", 0, pico_cnn::op_type::MaxPool, 5, 1, padding);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;
}

void TestPooling::runTestAvgPooling2d() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);

    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);

    fp_t input[25] = { 1,  2,  3,  4,  5,
                       6,  7,  8,  9, 10,
                       11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20,
                       21, 22, 23, 24, 25};

    fp_t expected_output[9] = {7, 8, 9,
                               12, 13, 14,
                               17, 18, 19};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, 3, 1, 0.0, nullptr, 1);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;
}

void TestPooling::runTestAvgPooling2dPadding() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);

    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);

    fp_t input[25] = {1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25};

    fp_t expected_output_count_include_pad_0[25] = {7, 7.5, 8, 8.5, 9,
                                                    9.5, 10, 10.5, 11, 11.5,
                                                    12, 12.5, 13, 13.5, 14,
                                                    14.5, 15, 15.5, 16, 16.5,
                                                    17, 17.5, 18, 18.5, 19};

    fp_t expected_output_count_include_pad_1[25] = {2.5200, 3.6000, 4.8000, 4.0800, 3.2400,
                                                    4.5600, 6.4000, 8.4000, 7.0400, 5.5200,
                                                    7.2000, 10.0000, 13.0000, 10.8000, 8.4000,
                                                    6.9600, 9.6000, 12.4000, 10.2400, 7.9200,
                                                    6.1200, 8.4000, 10.8000, 8.8800, 6.8400};

    fp_t expected_output_2_count_include_pad_0[25] = {4, 4.5, 5.5, 6.5, 7,
                                                      6.5, 7, 8, 9, 9.5,
                                                      11.5, 12, 13, 14, 14.5,
                                                      16.5, 17, 18, 19, 19.5,
                                                      19, 19.5, 20.5, 21.5, 22};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output_count_include_pad_0[i];
    }

    uint32_t padding[4] = {2, 2, 2, 2};
    uint32_t padding2[4] = {1, 1, 1, 1};

    /// Test 1: kernel = 5, padding = 2, count_include_pad = 0
    auto *layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, 5, 1, 0.0, padding, 0);
    layer->run(input_tensor, output_tensor);
    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    /// Test 2: kernel = 5, padding = 2, count_include_pad = 1
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output_count_include_pad_1[i];
    }

    layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, 5, 1, 0.0, padding, 1);
    layer->run(input_tensor, output_tensor);
    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    /// Test 3: kernel = 3, padding = 1, count_include_pad = 0
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output_2_count_include_pad_0[i];
    }

    layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, 3, 1, 0.0, padding2, 0);
    layer->run(input_tensor, output_tensor);
    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;
}
