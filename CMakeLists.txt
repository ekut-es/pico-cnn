cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

# Uncomment the following line and add jpeg to this list when you want to read a jpeg image as input
#find_package (JPEG REQUIRED)
list(APPEND LINK_LIBS m)

find_library(LIB_CPPUNIT cppunit)
if(NOT LIB_CPPUNIT)
    message(FATAL_ERROR "cppunit not found")
endif()


set(PICO_CNN_CPP_LIBRARY_SRCS
        ${PROJECT_SOURCE_DIR}/pico-cnn/tensor.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/layer.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/convolution.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/pooling/pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/pooling/max_pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/pooling/average_pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/pooling/global_max_pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/pooling/global_average_pooling.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/batch_normalization.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_functions/activation_function.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_functions/clip.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_functions/lrn.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_functions/relu.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_functions/sigmoid.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_functions/softmax.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/activation_functions/tan_h.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn/layers/fully_connected.cpp
)
set(PICO_CNN_CPP_IO_SRCS
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_binary_reference_data.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_binary_weights.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_imagenet_labels.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_imagenet_validation_labels.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_jpeg.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_means.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_mnist.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_pgm.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/write_float.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/write_pgm.cpp
)
add_library(pico-cnn ${PICO_CNN_CPP_LIBRARY_SRCS} ${PICO_CNN_CPP_IO_SRCS})
target_compile_options(pico-cnn PRIVATE -DDEBUG=0 -DINFO=1)

#add_executable(dummy_lenet ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/dummy_input.cpp
#                           ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/network.cpp
#)
#target_compile_options(dummy_lenet PRIVATE -DDEBUG=0 -DINFO=1)
#target_link_libraries(dummy_lenet pico-cnn ${LINK_LIBS})
#
#add_executable(reference_lenet ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/reference_input.cpp
#                               ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/network.cpp
#)
#target_compile_options(reference_lenet PRIVATE -DDEBUG=0 -DINFO=1)
#target_link_libraries(reference_lenet pico-cnn ${LINK_LIBS})

# Tests
set(UNIT_TESTS_SRCS
        ${PROJECT_SOURCE_DIR}/test/main.cpp
        ${PROJECT_SOURCE_DIR}/test/layers/test_tensor.cpp
        ${PROJECT_SOURCE_DIR}/test/layers/test_activation_functions.cpp
        ${PROJECT_SOURCE_DIR}/test/layers/test_fully_connected.cpp
        ${PROJECT_SOURCE_DIR}/test/layers/test_convolution.cpp
        ${PROJECT_SOURCE_DIR}/test/layers/test_pooling.cpp
        ${PROJECT_SOURCE_DIR}/test/layers/test_batch_normalization.cpp
        )
add_executable(unit_tests ${UNIT_TESTS_SRCS})

target_compile_options(unit_tests PRIVATE -DDEBUG=1 -DINFO=1)

target_include_directories(unit_tests PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(unit_tests cppunit pico-cnn ${LINK_LIBS})

enable_testing()

add_test(UnitTests unit_tests)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS unit_tests)

# removes pgm and float files
add_custom_target(remove_output_data
COMMAND rm ${CMAKE_BINARY_DIR}/*.pgm
COMMAND rm ${CMAKE_BINARY_DIR}/*.float
)