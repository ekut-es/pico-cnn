//
// Created by junga on 15.05.20.
//

#ifndef PICO_CNN_TEST_BATCH_NORMALIZATION_H
#define PICO_CNN_TEST_BATCH_NORMALIZATION_H

#include <cppunit/extensions/HelperMacros.h>
#include "../../pico-cnn/pico-cnn.h"


class TestBatchNormalization : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(TestBatchNormalization);
    CPPUNIT_TEST(runTestBatchNormalization_1);
    CPPUNIT_TEST(runTestBatchNormalization_2);
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp() override;
    void tearDown() override;

    void runTestBatchNormalization_1();
    void runTestBatchNormalization_2();
};


#endif //PICO_CNN_TEST_BATCH_NORMALIZATION_H
