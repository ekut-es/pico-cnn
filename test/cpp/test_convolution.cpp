#include "test_convolution.h"

CPPUNIT_TEST_SUITE_REGISTRATION(TestConvolution);

void TestConvolution::setUp() {
    TestFixture::setUp();

//    input_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
//    output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
//    expected_output_shape = new pico_cnn::naive::TensorShape(1, 4);
//    kernel_shape = new pico_cnn::naive::TensorShape(6, 4);
//    bias_shape = new pico_cnn::naive::TensorShape(1, 4);
//
//    input_tensor = new pico_cnn::naive::Tensor(input_shape);
//    output_tensor = new pico_cnn::naive::Tensor(output_shape);
//    expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
//    kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);
//    bias_tensor = new pico_cnn::naive::Tensor(bias_shape);
}

void TestConvolution::tearDown() {

//    delete input_tensor;
//    delete output_tensor;
//    delete expected_output_tensor;
//    delete kernel_tensor;
//    delete bias_tensor;
//
//    delete input_shape;
//    delete output_shape;
//    delete expected_output_shape;
//    delete kernel_shape;
//    delete bias_shape;

    TestFixture::tearDown();
}

void TestConvolution::runTestConvolution_0() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    uint32_t padding[4] = {1, 1, 1, 1};
    uint32_t stride[2] = {1, 1};
    uint32_t num_groups = 1;

    fp_t input[25] = {0., 1., 2., 3., 4.,
                      5., 6., 7., 8., 9.,
                      10., 11., 12., 13., 14.,
                      15., 16., 17., 18., 19.,
                      20., 21., 22., 23., 24.};

    fp_t expected_output[25] = {12., 21., 27., 33., 24.,
                                33., 54., 63., 72., 51.,
                                63., 99., 108., 117., 81.,
                                93., 144., 153., 162., 111.,
                                72., 111., 117., 123., 84.};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = 1.0;
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::Convolution("conv", 0, pico_cnn::op_type::Conv,
            kernel_tensor, nullptr, padding, stride, num_groups);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;


    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
    delete kernel_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;
    delete kernel_shape;
}

void TestConvolution::runTestConvolution_0_no_padding() {
    //PRINT_INFO("Test Convolution...")
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 5);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    //uint32_t padding[4] = {0, 0, 0, 0};
    uint32_t stride[2] = {1, 1};
    uint32_t num_groups = 1;

    fp_t input[25] = {0., 1., 2., 3., 4.,
                      5., 6., 7., 8., 9.,
                      10., 11., 12., 13., 14.,
                      15., 16., 17., 18., 19.,
                      20., 21., 22., 23., 24.};

    fp_t expected_output[9] = {54., 63., 72.,
                               99., 108., 117.,
                              144., 153., 162.};

    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = 1.0;
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::Convolution("conv", 0, pico_cnn::op_type::Conv,
                                                   kernel_tensor, nullptr, nullptr, stride, num_groups);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;


    delete input_tensor;
    delete output_tensor;
    delete expected_output_tensor;
    delete kernel_tensor;

    delete input_shape;
    delete output_shape;
    delete expected_output_shape;
    delete kernel_shape;
}

void TestConvolution::runTestConvolution_1() {

}

void TestConvolution::runTestConvolution_2() {

}

void TestConvolution::runTestConvolution_3() {

}

void TestConvolution::runTestConvolution_4() {

}

void TestConvolution::runTestConvolution_5() {

}
