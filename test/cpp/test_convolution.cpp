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

void TestConvolution::runTestConvolution_1d() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 1,11);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 1, 5);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1,1, 5);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 1, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    uint32_t stride[2] = {1, 2};
    uint32_t num_groups = 1;

    auto bias_shape = new pico_cnn::naive::TensorShape(1);
    auto bias_tensor = new pico_cnn::naive::Tensor(bias_shape);
    bias_tensor->access_blob(0) = -2;

    fp_t kernel[3] = {2, -1, 0};

    fp_t input[11] = {-9, 0, 3, -5, -10, 7, 3, -2, -2, 3,4};

    fp_t expected_output[5] = {-20, 9, -29, 6, -9};

    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = kernel[i];
    }
    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::Convolution("conv", 0, pico_cnn::op_type::Conv,
                                                   kernel_tensor, bias_tensor, nullptr, stride, num_groups);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;


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
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 4, 3);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 4, 3);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 4, 3);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    uint32_t padding[4] = {1, 1, 1, 1};
    uint32_t stride[2] = {1, 1};
    uint32_t num_groups = 1;

    fp_t kernel[9] = {-5, -1, -3,
                      10, 3,  0,
                      4, -6,  6};

    fp_t input[12] = {-5,-4, -5,
                      -6, 5, -10,
                      8, 9, -3,
                      0,-9, -1};

    fp_t expected_output[12] = {51,-176,  25,
                                5, -41,  99,
                                -39, 210,  36,
                                -35, -67,-135};

    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = kernel[i];
    }
    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
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

void TestConvolution::runTestConvolution_2() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 4, 3);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 4, 3);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 4, 3);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    uint32_t padding[4] = {1, 1, 1, 1};
    uint32_t stride[2] = {1, 1};
    uint32_t num_groups = 1;

    fp_t kernel[9] = {0,0,0,
                      0,0,0,
                      0,0,0};

    fp_t input[12] = {-5,-4, -5,
                      -6, 5, -10,
                      8, 9, -3,
                      0,-9, -1};

    fp_t expected_output[12] = {0,0,0,
                                0,0,0,
                                0,0,0,
                                0,0,0};

    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = kernel[i];
    }
    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
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

void TestConvolution::runTestConvolution_3() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 8, 6);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 10, 8);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 10, 8);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 1, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    uint32_t padding[4] = {0,1,2,3};
    uint32_t stride[2] = {1, 1};
    uint32_t num_groups = 1;

    fp_t kernel[3] = {-3, -1, 0};

    fp_t input[48] = {8,  9,   9,  6, -10, 10,
                      10, -8,   2,  6,  -3, -8,
                      9,  0,   7, 10,  -7,  5,
                      -8,  2,  -8,  9,  -3, -5,
                      -2,  5,  -4, -1,   5, -1,
                      -7, -6,   1, -8,   9, -6,
                      -2, -4, -10,  6, -10, -2,
                      2,  5,   1, -7,   3, -1};

    fp_t expected_output[80] = {  -8, -33, -36, -33,  -8,  20, -30, 0,
                                  -10, -22,  22, -12, -15,  17,  24, 0,
                                  -9, -27,  -7, -31, -23,  16,  -15, 0,
                                  8,  22,   2,  15, -24,  14, 15, 0,
                                  2,   1, -11,  13,  -2, -14, 3, 0,
                                  7,  27,  17,   5,  15, -21, 18, 0,
                                  2,  10,  22,  24,  -8,  32, 6, 0,
                                  -2, -11, -16,   4,  18,  -8, 3, 0,
                                  0,   0,   0,   0,   0,   0,  0, 0,
                                  0,   0,   0,   0,   0,   0,  0, 0};

    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = kernel[i];
    }
    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
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

void TestConvolution::runTestConvolution_4() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 8, 6);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 3);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 3);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 1, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    uint32_t padding[4] = {0,1,2,3};
    uint32_t stride[2] = {2, 3};
    uint32_t num_groups = 1;

    fp_t kernel[3] = {-3, -1, 0};

    fp_t input[48] = {8,  9,   9,  6, -10, 10,
                      10, -8,   2,  6,  -3, -8,
                      9,  0,   7, 10,  -7,  5,
                      -8,  2,  -8,  9,  -3, -5,
                      -2,  5,  -4, -1,   5, -1,
                      -7, -6,   1, -8,   9, -6,
                      -2, -4, -10,  6, -10, -2,
                      2,  5,   1, -7,   3, -1};

    fp_t expected_output[15] = {  -8, -33, -30,
                                  -9, -31, -15,
                                  2,  13,   3,
                                  2,  24,   6,
                                  0,   0,   0};

    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = kernel[i];
    }
    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
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

void TestConvolution::runTestConvolution_5() {
    auto input_shape = new pico_cnn::naive::TensorShape(1, 1, 7, 5);
    auto output_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);
    auto expected_output_shape = new pico_cnn::naive::TensorShape(1, 1, 3, 3);
    auto kernel_shape = new pico_cnn::naive::TensorShape(1, 1, 5, 3);


    auto input_tensor = new pico_cnn::naive::Tensor(input_shape);
    auto output_tensor = new pico_cnn::naive::Tensor(output_shape);
    auto expected_output_tensor = new pico_cnn::naive::Tensor(expected_output_shape);
    auto kernel_tensor = new pico_cnn::naive::Tensor(kernel_shape);

    uint32_t padding[4] = {1, 2, 4, 0};
    uint32_t stride[2] = {3, 2};
    uint32_t num_groups = 1;

    auto bias_shape = new pico_cnn::naive::TensorShape(1);
    auto bias_tensor = new pico_cnn::naive::Tensor(bias_shape);
    bias_tensor->access_blob(0) = 0.4;

    fp_t kernel[15] = {-3, -4,  1,
                      -5,  5,  2,
                      1,  2, -1,
                      -1,  1, -4,
                      -4, -2,  5};

    fp_t input[35] = {4, 13,  13,  -2,   6,
                      -7, 15, -11,  -9, -15,
                      -8, -3,   2,   6,  -9,
                      1, -8,  13,  -6,  -7,
                      -6, -6,  13,  13,   8,
                      -15, 14, -12,   8,   1,
                      3,  6,   7, -12,   2};

    fp_t expected_output[9] = { 52.4, 179.4, -111.6,
                                75.4, 76.4, -94.6,
                                -8.6, 6.4, -85.6};

    for(uint32_t i = 0; i < kernel_tensor->num_elements(); i++) {
        kernel_tensor->access_blob(i) = kernel[i];
    }
    for(uint32_t i = 0; i < input_tensor->num_elements(); i++) {
        input_tensor->access_blob(i) = input[i];
    }
    for(uint32_t i = 0; i < expected_output_tensor->num_elements(); i++) {
        expected_output_tensor->access_blob(i) = expected_output[i];
    }

    auto *layer = new pico_cnn::naive::Convolution("conv", 0, pico_cnn::op_type::Conv,
                                                   kernel_tensor, bias_tensor, padding, stride, num_groups);
    layer->run(input_tensor, output_tensor);

    CPPUNIT_ASSERT(*output_tensor == *expected_output_tensor);

    delete layer;


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
}
