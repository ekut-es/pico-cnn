cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
list(APPEND LINK_LIBS m)

find_library(LIB_CPPUNIT cppunit)
if(NOT LIB_CPPUNIT)
    message(FATAL_ERROR "cppunit not found")
endif()


set(PICO_CNN_CPP_LIBRARY_SRCS
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/tensor_shape.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/tensor.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/layer.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/convolution.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_function.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/fully_connected.cpp
)
add_library(pico-cnn-cpp ${PICO_CNN_CPP_LIBRARY_SRCS})

# Tests
set(UNIT_TESTS_SRCS
        ${PROJECT_SOURCE_DIR}/test/cpp/main.cpp
        ${PROJECT_SOURCE_DIR}/test/cpp/test_tensor.cpp
        ${PROJECT_SOURCE_DIR}/test/cpp/test_activation_functions.cpp
        ${PROJECT_SOURCE_DIR}/test/cpp/test_utility.cpp
)
add_executable(unit_tests ${UNIT_TESTS_SRCS})

target_compile_options(unit_tests PRIVATE -DDEBUG=1 -DINFO=1)

target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(unit_tests cppunit pico-cnn-cpp)

enable_testing()

add_test(UnitTests unit_tests)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS unit_tests)

# removes pgm and float files
add_custom_target(remove_output_data
COMMAND rm ${CMAKE_BINARY_DIR}/*.pgm
COMMAND rm ${CMAKE_BINARY_DIR}/*.float
)