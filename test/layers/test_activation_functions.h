/**
 * @brief Tests covering pico_cnn::naive::ActivationFunctions
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_TEST_ACTIVATION_FUNCTIONS_H
#define PICO_CNN_TEST_ACTIVATION_FUNCTIONS_H

#include <cppunit/extensions/HelperMacros.h>
#include "../../pico-cnn/pico-cnn.h"
#include "../../pico-cnn/utils.h"

class TestActivationFunctions : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(TestActivationFunctions);
    CPPUNIT_TEST(runTestActivationFunction);
    CPPUNIT_TEST(runTestClip);
    CPPUNIT_TEST(runTestLRN);
    CPPUNIT_TEST(runTestReLU);
    CPPUNIT_TEST(runTestLeakyReLU);
    CPPUNIT_TEST(runTestParameterizedReLU);
    CPPUNIT_TEST(runTestSigmoid);
    CPPUNIT_TEST(runTestSoftmax);
    CPPUNIT_TEST(runTestTanH);
    CPPUNIT_TEST_SUITE_END();

private:
    pico_cnn::naive::Tensor *input_tensor, *output_tensor, *expected_output_tensor;


public:
    void setUp() override;
    void tearDown() override;

    void runTestActivationFunction();
    void runTestClip();
    void runTestLRN();
    void runTestReLU();
    void runTestLeakyReLU();
    void runTestParameterizedReLU();
    void runTestSigmoid();
    void runTestSoftmax();
    void runTestTanH();
};


#endif //PICO_CNN_TEST_ACTIVATION_FUNCTIONS_H
