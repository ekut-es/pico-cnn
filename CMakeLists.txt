cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
list(APPEND LINK_LIBS m)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O1")

find_library(LIB_CPPUNIT cppunit)
if(NOT LIB_CPPUNIT)
    message(FATAL_ERROR "cppunit not found")
endif()


set(PICO_CNN_CPP_LIBRARY_SRCS
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/tensor.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/layer.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/convolution.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/pooling/pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/pooling/max_pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/pooling/average_pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/pooling/global_max_pooling.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/pooling/global_average_pooling.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/batch_normalization.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_functions/activation_function.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_functions/clip.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_functions/lrn.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_functions/relu.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_functions/sigmoid.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_functions/softmax.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/activation_functions/tan_h.cpp

        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/layers/fully_connected.cpp
)
set(PICO_CNN_CPP_IO_SRCS
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/io/read_binary_reference_data.cpp
        ${PROJECT_SOURCE_DIR}/pico-cnn-cpp/io/read_binary_weights.cpp
)
add_library(pico-cnn-cpp ${PICO_CNN_CPP_LIBRARY_SRCS} ${PICO_CNN_CPP_IO_SRCS})
target_compile_options(pico-cnn-cpp PRIVATE ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -DDEBUG=0 -DINFO=1)

#add_executable(dummy_lenet ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/dummy_input.cpp
#                           ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/network.cpp
#)
#target_compile_options(dummy_lenet PRIVATE -DDEBUG=0 -DINFO=1)
#target_link_libraries(dummy_lenet pico-cnn-cpp)
#
#add_executable(reference_lenet ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/reference_input.cpp
#                               ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet/network.cpp
#)
#target_compile_options(reference_lenet PRIVATE -DDEBUG=0 -DINFO=1)
#target_link_libraries(reference_lenet pico-cnn-cpp)

add_executable(reference_inception ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/reference_input.cpp
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network.cpp
        )
target_compile_options(reference_inception PRIVATE -DDEBUG=0 -DINFO=1)
target_link_libraries(reference_inception pico-cnn-cpp)

add_executable(dummy_inception ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/dummy_input.cpp
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network.cpp)
target_compile_options(dummy_inception PRIVATE  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -DDEBUG=0 -DINFO=1)
target_link_libraries(dummy_inception pico-cnn-cpp)

# Tests
set(UNIT_TESTS_SRCS
        ${PROJECT_SOURCE_DIR}/test/cpp/main.cpp
        ${PROJECT_SOURCE_DIR}/test/cpp/test_tensor.cpp
        ${PROJECT_SOURCE_DIR}/test/cpp/test_activation_functions.cpp
        ${PROJECT_SOURCE_DIR}/test/cpp/test_fully_connected.cpp
        ${PROJECT_SOURCE_DIR}/test/cpp/test_convolution.cpp
#        ${PROJECT_SOURCE_DIR}/test/cpp/test_pooling.cpp
#        ${PROJECT_SOURCE_DIR}/test/cpp/test_batch_normalization.cpp
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