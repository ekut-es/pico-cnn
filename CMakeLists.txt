cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

option(BUILD_UTILS "Utility programs from the 'util' directory will be built." OFF)

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
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_binary_sample_data.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_binary_weights.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_imagenet_labels.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_jpeg.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_means.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_pgm.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/read_weights.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/write_float.c
        ${PROJECT_SOURCE_DIR}/pico-cnn/io/write_pgm.c
        pico-cnn/layers/batch_normalization.c pico-cnn/layers/batch_normalization.h)
add_library(libpico-cnn ${PICO_CNN_LIBRARY_SRCS})

if(BUILD_UTILS)
    add_executable(compare_float_files ${PROJECT_SOURCE_DIR}/util/compare_float_files.c)

    add_executable(display_pgm_cmd ${PROJECT_SOURCE_DIR}/util/display_pgm_cmd.c)
    target_link_libraries(display_pgm_cmd ${LINK_LIBS})

    add_executable(jpeg_to_rgb_pgms ${PROJECT_SOURCE_DIR}/util/jpeg_to_rgb_pgms.c)
	target_link_libraries(jpeg_to_rgb_pgms ${LINK_LIBS} jpeg)

    add_executable(find_max_min_weight ${PROJECT_SOURCE_DIR}/util/find_max_min_weight.c)
endif()

#add_executable(ekut-raw-padding ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/ekut-raw-padding/dummy_input.c
#        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/ekut-raw-padding/network.c
#        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/ekut-raw-padding/network_initialization.c
#        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/ekut-raw-padding/network_cleanup.c)
#target_link_libraries(ekut-raw-padding libpico-cnn ${LINK_LIBS})

#add_executable(lenet_avg_dummy ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet_avg/dummy_input.c
#        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet_avg/network.c
#        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet_avg/network_initialization.c
#        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet_avg/network_cleanup.c)
#target_link_libraries(lenet_avg_dummy libpico-cnn ${LINK_LIBS})

add_executable(mobilenet_reference ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/reference_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_cleanup.c)
target_link_libraries(mobilenet_reference libpico-cnn ${LINK_LIBS})

add_executable(mobilenet_dummy ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/dummy_input.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_initialization.c
        ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mobilenetv2-1/network_cleanup.c)
target_link_libraries(mobilenet_dummy libpico-cnn ${LINK_LIBS})

add_executable(run_tests ${PROJECT_SOURCE_DIR}/test/run_tests.c

        ${PROJECT_SOURCE_DIR}/test/layers/test_activation_function.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_batch_normalization.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_convolution.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_fully_connected.c
        ${PROJECT_SOURCE_DIR}/test/layers/test_pooling.c
        ${PROJECT_SOURCE_DIR}/test/utility_functions.c
  )
>>>>>>> ddb64c467f9bc9bb81644d11af4da601e23ebc03
target_link_libraries(run_tests libpico-cnn ${LINK_LIBS})

# removes pgm and float files
add_custom_target(remove_output_data
    COMMAND rm ${CMAKE_BINARY_DIR}/*.pgm
    COMMAND rm ${CMAKE_BINARY_DIR}/*.float
)

enable_testing()
add_test(TestPooling run_tests)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS run_tests)
