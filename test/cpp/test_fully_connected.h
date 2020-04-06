//
// Created by junga on 06.04.20.
//

#ifndef PICO_CNN_TEST_FULLY_CONNECTED_H
#define PICO_CNN_TEST_FULLY_CONNECTED_H

#include <cppunit/extensions/HelperMacros.h>
#include "../../pico-cnn-cpp/pico-cnn.h"
#include "../../pico-cnn-cpp/utils.h"

class TestFullyConnected : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(TestFullyConnected);
    CPPUNIT_TEST(runTestFullyConnected);
    CPPUNIT_TEST_SUITE_END();

private:
    pico_cnn::naive::TensorShape *input_shape, *output_shape, *expected_output_shape, *kernel_shape, *bias_shape;
    pico_cnn::naive::Tensor *input_tensor, *output_tensor, *expected_output_tensor, *kernel_tensor, *bias_tensor;

public:
    void setUp() override;
    void tearDown() override;

    void runTestFullyConnected();
};


#endif //PICO_CNN_TEST_FULLY_CONNECTED_H
