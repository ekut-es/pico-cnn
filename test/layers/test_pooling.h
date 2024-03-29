//
// Created by junga on 17.04.20.
//

#ifndef PICO_CNN_TEST_POOLING_H
#define PICO_CNN_TEST_POOLING_H

#include <cppunit/extensions/HelperMacros.h>
#include "../../pico-cnn/pico-cnn.h"
#include "../../pico-cnn/utils.h"

class TestPooling : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(TestPooling);
    CPPUNIT_TEST(runTestMaxPooling1d);
    CPPUNIT_TEST(runTestMaxPooling1dPadding);
    CPPUNIT_TEST(runTestMaxPooling2d);
    CPPUNIT_TEST(runTestMaxPooling2dPadding);
    CPPUNIT_TEST(runTestAvgPooling1d);
    CPPUNIT_TEST(runTestAvgPooling1dPadding);
    CPPUNIT_TEST(runTestAvgPooling2d);
    CPPUNIT_TEST(runTestAvgPooling2dPadding);
    CPPUNIT_TEST(runTestGlobalAvgPool2d);
    CPPUNIT_TEST(runTestGlobalMaxPool2d);
    CPPUNIT_TEST_SUITE_END();

private:

public:
    void setUp() override;
    void tearDown() override;

    void runTestMaxPooling1d();
    void runTestMaxPooling1dPadding();

    void runTestMaxPooling2d();
    void runTestMaxPooling2dPadding();

    void runTestAvgPooling1d();
    void runTestAvgPooling1dPadding();

    void runTestAvgPooling2d();
    void runTestAvgPooling2dPadding();

    void runTestGlobalAvgPool2d();
    void runTestGlobalMaxPool2d();
};


#endif //PICO_CNN_TEST_POOLING_H
