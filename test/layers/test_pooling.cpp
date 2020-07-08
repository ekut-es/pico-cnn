#include "test_pooling.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestPooling);

void TestPooling::setUp() {
    TestFixture::setUp();
}

void TestPooling::tearDown() {
    TestFixture::tearDown();
}

void TestPooling::runTestMaxPooling1d() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 16);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 7);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 7);

    fp_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    fp_t expected_output[7] = {3, 5, 7, 9, 11, 13, 15};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    uint32_t kernel_size[1] = {3};
    uint32_t stride[1] = {2};

    auto *layer = new pico_cnn::naive::MaxPooling("MaxPool", 0, pico_cnn::op_type::MaxPool, kernel_size, stride, nullptr);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestMaxPooling1dPadding() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 16);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 8);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 8);

    fp_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    fp_t expected_output[8] = {2, 4, 6, 8, 10, 12, 14, 16};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    uint32_t kernel_size[2] = {1, 3};
    uint32_t stride[2] = {1, 2};
    uint32_t padding[4] = {0, 1, 0, 1};

    auto *layer = new pico_cnn::naive::MaxPooling("MaxPool", 0, pico_cnn::op_type::MaxPool, kernel_size, stride, padding);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestMaxPooling2d() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 4, 4);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 2, 2);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 2, 2);

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

    uint32_t kernel_size[2] = {3, 3};
    uint32_t stride[2] = {1, 1};

    auto *layer = new pico_cnn::naive::MaxPooling("MaxPool", 0, pico_cnn::op_type::MaxPool, kernel_size, stride, nullptr);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestMaxPooling2dPadding() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 5, 5);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 5, 5);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 5, 5);

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

    uint32_t kernel_size[2] = {5, 5};
    uint32_t stride[2] = {1, 1};
    uint32_t padding[4] = {2, 2, 2, 2};

    auto *layer = new pico_cnn::naive::MaxPooling("MaxPool", 0, pico_cnn::op_type::MaxPool, kernel_size, stride, padding);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestAvgPooling1d() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 10);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 8);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 8);

    fp_t input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    fp_t expected_output[8] = {2, 3, 4, 5, 6, 7, 8, 9};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    uint32_t kernel_size[2] = {3};
    uint32_t stride[2] = {1};

    auto *layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, kernel_size, stride, nullptr, 1);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestAvgPooling1dPadding() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 10);
    auto expected_output_tensor_count_include_pad_0 = new pico_cnn::naive::Tensor(1, 1, 10);
    auto expected_output_tensor_count_include_pad_1 = new pico_cnn::naive::Tensor(1, 1, 10);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 10);

    fp_t input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    fp_t expected_output_count_include_pad_0[10] = {2, 2.5, 3, 4, 5, 6, 7, 8, 8.5, 9};

    fp_t expected_output_count_include_pad_1[10] = {1.2, 2, 3, 4, 5, 6, 7, 8, 6.8, 5.4};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor_count_include_pad_0->num_elements(); i++) {
        expected_output_tensor_count_include_pad_0->access_blob(i) = expected_output_count_include_pad_0[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor_count_include_pad_1->num_elements(); i++) {
        expected_output_tensor_count_include_pad_1->access_blob(i) = expected_output_count_include_pad_1[i];
    }

    uint32_t kernel_size[2] = {5};
    uint32_t stride[2] = {1};
    uint32_t padding[2] = {2, 2};

    auto *layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, kernel_size, stride, padding, 0);
    auto *layer2 = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, kernel_size, stride, padding, 1);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor_count_include_pad_0);

    layer2->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor_count_include_pad_1);

    delete layer;
    delete layer2;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor_count_include_pad_0;
    delete expected_output_tensor_count_include_pad_1;
}

void TestPooling::runTestAvgPooling2d() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 5, 5);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 3, 3);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 3, 3);

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

    uint32_t kernel_size[2] = {3, 3};
    uint32_t stride[2] = {1, 1};

    auto *layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, kernel_size, stride, nullptr, 1);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestAvgPooling2dPadding() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 5, 5);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 5, 5);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 5, 5);

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

    uint32_t kernel_size[2] = {5, 5};
    uint32_t kernel_size2[2] = {3, 3};
    uint32_t stride[2] = {1, 1};
    uint32_t padding[4] = {2, 2, 2, 2};
    uint32_t padding2[4] = {1, 1, 1, 1};

    /// Test 1: kernel = 5, padding = 2, count_include_pad = 0
    auto *layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, kernel_size, stride, padding, 0);
    layer->run(input_tensor, output_tensor);
    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    /// Test 2: kernel = 5, padding = 2, count_include_pad = 1
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output_count_include_pad_1[i];
    }

    layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, kernel_size, stride, padding, 1);
    layer->run(input_tensor, output_tensor);
    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    /// Test 3: kernel = 3, padding = 1, count_include_pad = 0
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output_2_count_include_pad_0[i];
    }

    layer = new pico_cnn::naive::AveragePooling("AvgPool", 0, pico_cnn::op_type::AveragePool, kernel_size2, stride, padding2, 0);
    layer->run(input_tensor, output_tensor);
    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestGlobalAvgPool2d() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 3, 4);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 1);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 1);

    fp_t input[20] = {-7, 13, -7, -8,
                      -15, -4, -7,  4,
                      6, -6, -6,  6};

    fp_t expected_output[1] = {-2.5833333};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::GlobalAveragePooling("GlobAvgPool", 0, pico_cnn::op_type::GlobalAveragePool, nullptr, nullptr, nullptr);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}

void TestPooling::runTestGlobalMaxPool2d() {

    auto input_tensor = new pico_cnn::naive::Tensor(1, 1, 4, 5);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 1);
    auto output_tensor = new pico_cnn::naive::Tensor(1, 1, 1, 1);

    fp_t input[20] = {17,  17, -15,  8, -15,
                      2,  13,  -2, -5,  6,
                      9, -17,  18, 20, -5,
                      -14,   4,  11,  7, 18};

    fp_t expected_output[1] = {20};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::GlobalMaxPooling("GlobMaxPool", 0, pico_cnn::op_type::GlobalMaxPool, nullptr, nullptr, nullptr);

    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;

    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
}
