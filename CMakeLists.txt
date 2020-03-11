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

# MobileNet
add_executable(mobilenet_dummy ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/dummy_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_cleanup.c)
target_link_libraries(mobilenet_dummy libpico-cnn ${LINK_LIBS})

add_executable(mobilenet_reference ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/reference_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_cleanup.c)
target_link_libraries(mobilenet_reference libpico-cnn ${LINK_LIBS})

# Inception
add_executable(inceptionv3_dummy ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/dummy_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network_cleanup.c)
target_link_libraries(inceptionv3_dummy libpico-cnn ${LINK_LIBS})

add_executable(inceptionv3_reference ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/reference_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inceptionv3/network_cleanup.c)
target_link_libraries(inceptionv3_reference libpico-cnn ${LINK_LIBS})

# InceptionResNet
add_executable(inception_resnet_dummy ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/dummy_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/network_cleanup.c)
target_link_libraries(inception_resnet_dummy libpico-cnn ${LINK_LIBS})

add_executable(inception_resnet_reference ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/reference_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/inception_resnet_v2/network_cleanup.c)
target_link_libraries(inception_resnet_reference libpico-cnn ${LINK_LIBS})

# TC-ResNet
add_executable(tc_resnet_dummy ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/tc-res8-update/dummy_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/tc-res8-update/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/tc-res8-update/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/tc-res8-update/network_cleanup.c)
target_link_libraries(tc_resnet_dummy libpico-cnn ${LINK_LIBS})

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
