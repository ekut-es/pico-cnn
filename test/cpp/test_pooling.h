//
// Created by junga on 17.04.20.
//

#ifndef PICO_CNN_TEST_POOLING_H
#define PICO_CNN_TEST_POOLING_H

#include <cppunit/extensions/HelperMacros.h>
#include "../../pico-cnn-cpp/pico-cnn.h"
#include "../../pico-cnn-cpp/utils.h"

class TestPooling : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(TestPooling);
    CPPUNIT_TEST(runTestMaxPooling2d);
    CPPUNIT_TEST(runTestMaxPooling2dPadding);
    CPPUNIT_TEST(runTestAvgPooling2d);
    CPPUNIT_TEST(runTestAvgPooling2dPadding);
    CPPUNIT_TEST_SUITE_END();

private:

public:
    void setUp() override;
    void tearDown() override;

    void runTestMaxPooling2d();
    void runTestMaxPooling2dPadding();

    void runTestAvgPooling2d();
    void runTestAvgPooling2dPadding();
};


#endif //PICO_CNN_TEST_POOLING_H
