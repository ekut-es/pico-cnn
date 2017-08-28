cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

option(BUILD_TESTS "Test programs from the 'test' directory will be built." OFF)
option(BUILD_UTILS "Utility programs from the 'util' directory will be built." OFF)
option(BUILD_EXAMPLES "Example programs from the 'examples' directory will be built." OFF)

set(CMAKE_C_FLAGS "-Wall")
set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

if(BUILD_TESTS)
    add_executable(max_pooling_test ${PROJECT_SOURCE_DIR}/test/max_pooling_test.c)
    target_link_libraries(max_pooling_test m)
endif()

if(BUILD_UTILS)
    add_executable(compare_float_files ${PROJECT_SOURCE_DIR}/util/compare_float_files.c)
endif()

if(BUILD_EXAMPLES)
    add_executable(lenet_tanh ${PROJECT_SOURCE_DIR}/examples/lenet_tanh.c)
    target_link_libraries(lenet_tanh m)

    add_executable(lenet_caffe ${PROJECT_SOURCE_DIR}/examples/lenet_caffe.c)
    target_link_libraries(lenet_caffe m)
endif()
