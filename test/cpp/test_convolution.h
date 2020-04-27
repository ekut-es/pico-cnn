//
// Created by junga on 07.04.20.
//

#ifndef PICO_CNN_TEST_CONVOLUTION_H
#define PICO_CNN_TEST_CONVOLUTION_H

#include <cppunit/extensions/HelperMacros.h>
#include "../../pico-cnn-cpp/pico-cnn.h"
#include "../../pico-cnn-cpp/utils.h"

class TestConvolution : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(TestConvolution);
    CPPUNIT_TEST(runTestConvolution_1d);
    CPPUNIT_TEST(runTestConvolution_0);
    CPPUNIT_TEST(runTestConvolution_0_no_padding);
    CPPUNIT_TEST(runTestConvolution_1);
    CPPUNIT_TEST(runTestConvolution_2);
    CPPUNIT_TEST(runTestConvolution_3);
    CPPUNIT_TEST(runTestConvolution_4);
    CPPUNIT_TEST(runTestConvolution_5);
    CPPUNIT_TEST(runTestConvolution_6);
    CPPUNIT_TEST(runTestConvolution_7);
    CPPUNIT_TEST_SUITE_END();

private:
    //pico_cnn::naive::TensorShape *input_shape, *output_shape, *expected_output_shape, *kernel_shape, *bias_shape;
    //pico_cnn::naive::Tensor *input_tensor, *output_tensor, *expected_output_tensor, *kernel_tensor, *bias_tensor;

public:
    void setUp() override;
    void tearDown() override;

    void runTestConvolution_1d();
    void runTestConvolution_0();
    void runTestConvolution_0_no_padding();
    void runTestConvolution_1();
    void runTestConvolution_2();
    void runTestConvolution_3();
    void runTestConvolution_4();
    void runTestConvolution_5();
    void runTestConvolution_6();
    void runTestConvolution_7();

};


#endif //PICO_CNN_TEST_CONVOLUTION_H
