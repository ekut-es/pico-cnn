/**
 * @brief contains all activation functions
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 * @author Nils Weinhardt (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef TEST_FULLY_CONNECTED_H
#define TEST_FULLY_CONNECTED_H

#include "pico-cnn/parameters.h"
#include "pico-cnn/layers/fully_connected.h"
#include "../utility_functions.h" // for floatsAlmostEqual()

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int32_t test_fully_connected();

#endif // TEST_FULLY_CONNECTED_H
