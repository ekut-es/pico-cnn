cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
list(APPEND LINK_LIBS m)

set(PICO_CNN_LIBRARY_SRCS
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/convolution.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/pooling.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_function.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/fully_connected.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/concatenate.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/batch_normalization.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_binary_reference_data.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_binary_weights.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_imagenet_labels.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_jpeg.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_means.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_pgm.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/write_float.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/write_pgm.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/utils.c
)
add_library(libpico-cnn ${PICO_CNN_LIBRARY_SRCS})

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

add_executable(cpp_test ${PROJECT_SOURCE_DIR}/test/cpp/main.cpp)
target_link_libraries(cpp_test pico-cnn-cpp)

# Tests
add_executable(run_tests ${PROJECT_SOURCE_DIR}/test/run_tests.c

        ${PROJECT_SOURCE_DIR}/test/layers/test_activation_function.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_batch_normalization.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_convolution.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_fully_connected.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_pooling.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_concatenate.c
        ${PROJECT_SOURCE_DIR}/test/utility_functions.c
  )
target_link_libraries(run_tests libpico-cnn ${LINK_LIBS})

# removes pgm and float files
add_custom_target(remove_output_data
    COMMAND rm ${CMAKE_BINARY_DIR}/*.pgm
    COMMAND rm ${CMAKE_BINARY_DIR}/*.float
)

enable_testing()
add_test(TestPooling run_tests)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS run_tests)
