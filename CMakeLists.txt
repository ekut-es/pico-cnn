cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

option(BUILD_TESTS "Test programs from the 'test' directory will be built." OFF)
option(BUILD_UTILS "Utility programs from the 'util' directory will be built." OFF)
option(BUILD_EXAMPLES "Example programs from the 'examples' directory will be built." OFF)
option(BUILD_ARM_EXAMPLES "ARM example programs from the 'examples' directory will be built." OFF)

set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
list(APPEND LINK_LIBS m)
    
if(BUILD_TESTS)
    add_executable(preprocessor_macro_test ${PROJECT_SOURCE_DIR}/util/preprocessor_macro_test.c)
endif()

if(BUILD_UTILS)
    add_executable(compare_float_files ${PROJECT_SOURCE_DIR}/util/compare_float_files.c)

    add_executable(display_pgm_cmd ${PROJECT_SOURCE_DIR}/util/display_pgm_cmd.c)
    target_link_libraries(display_pgm_cmd ${LINK_LIBS})

    add_executable(jpeg_to_rgb_pgms ${PROJECT_SOURCE_DIR}/util/jpeg_to_rgb_pgms.c)
	target_link_libraries(jpeg_to_rgb_pgms ${LINK_LIBS} jpeg)

    add_executable(find_max_min_weight ${PROJECT_SOURCE_DIR}/util/find_max_min_weight.c)
endif()

if(BUILD_EXAMPLES)
    add_executable(caffe_lenet_naive ${PROJECT_SOURCE_DIR}/examples/caffe_lenet_naive.c)
    target_link_libraries(caffe_lenet_naive ${LINK_LIBS})

    add_executable(caffe_lenet_fixed16 ${PROJECT_SOURCE_DIR}/examples/caffe_lenet_fixed16.c)
    target_link_libraries(caffe_lenet_fixed16 ${LINK_LIBS})

    add_executable(ekut_es_alexnet_naive ${PROJECT_SOURCE_DIR}/examples/ekut_es_alexnet_naive.c)
    target_link_libraries(ekut_es_alexnet_naive ${LINK_LIBS} jpeg)
endif()

if(BUILD_ARM_EXAMPLES)
    add_executable(caffe_lenet_arm_cpu ${PROJECT_SOURCE_DIR}/examples/caffe_lenet_arm_cpu.c)
    target_compile_options(caffe_lenet_arm_cpu PRIVATE "-mfpu=neon")
    target_link_libraries(caffe_lenet_arm_cpu ${LINK_LIBS})

    add_executable(ekut_es_alexnet_arm_cpu ${PROJECT_SOURCE_DIR}/examples/ekut_es_alexnet_arm_cpu.c)
    target_compile_options(ekut_es_alexnet_arm_cpu PRIVATE "-mfpu=neon")
    target_link_libraries(ekut_es_alexnet_arm_cpu ${LINK_LIBS} jpeg)
endif()

add_executable(clion_lenet ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/lenet_1_1_28_28/lenet_1_1_28_28.c)
target_link_libraries(clion_lenet ${LINK_LIBS})

add_executable(alexnet_trained ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/alexnet_trained/alexnet_trained.c)
target_link_libraries(alexnet_trained ${LINK_LIBS} jpeg)

add_executable(clion_alexnet ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/alexnet/reference_input.c)
target_link_libraries(clion_alexnet ${LINK_LIBS})

add_executable(clion_mnist_mlp ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mnist_mlp/mnist_mlp.c)
target_link_libraries(clion_mnist_mlp ${LINK_LIBS})

add_executable(clion_mnist_p_1 ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mnist_simple_mlp_batch1/mnist_simple_mlp_batch1.c)
target_link_libraries(clion_mnist_p_1 ${LINK_LIBS})

add_executable(clion_mnist_p_50 ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/mnist_simple_mlp_batch50/mnist_simple_mlp_batch50.c)
target_link_libraries(clion_mnist_p_50 ${LINK_LIBS})

add_executable(clion_vgg16 ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/vgg16/vgg16.c)
target_link_libraries(clion_vgg16 ${LINK_LIBS} jpeg)

add_executable(clion_ekut-raw-cnn6-relu ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/ekut-raw-cnn6-relu/reference_input.c)
target_link_libraries(clion_ekut-raw-cnn6-relu ${LINK_LIBS})

add_executable(debug-max-pool ${PROJECT_SOURCE_DIR}/onnx_import/generated_code/alexnet/dummy_input.c)
target_link_libraries(debug-max-pool ${LINK_LIBS})

# removes pgm and float files
add_custom_target(remove_output_data
    COMMAND rm ${CMAKE_BINARY_DIR}/*.pgm
    COMMAND rm ${CMAKE_BINARY_DIR}/*.float
)
