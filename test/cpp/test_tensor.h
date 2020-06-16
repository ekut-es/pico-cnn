/**
 * @brief Tests covering pico_cnn::naive::TensorShape and pico_cnn::naive::Tensor
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_TEST_TENSOR_H
#define PICO_CNN_TEST_TENSOR_H

#include <cppunit/extensions/HelperMacros.h>
#include "../../pico-cnn-cpp/pico-cnn.h"

class TestTensor : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(TestTensor);
    CPPUNIT_TEST(runTestTensorAccess);
    CPPUNIT_TEST(runTestTensorAddition);
    CPPUNIT_TEST(runTestTensorGetPtr);
    CPPUNIT_TEST(runTestTensorExpandPadding);
    CPPUNIT_TEST(runTestTensorConcatDim0);
    CPPUNIT_TEST_SUITE_END();

private:
    pico_cnn::naive::Tensor *tensor1, *tensor2, *tensor3, *tensor4, *tensor5, *tensor6;


public:
    void setUp() override;
    void tearDown() override;
    void runTestTensorAccess();
    void runTestTensorAddition();
    void runTestTensorGetPtr();
    void runTestTensorExpandPadding();
    void runTestTensorConcatDim0();

};


#endif //PICO_CNN_TEST_TENSOR_H
