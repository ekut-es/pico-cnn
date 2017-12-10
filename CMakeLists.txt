cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

option(BUILD_TESTS "Test programs from the 'test' directory will be built." OFF)
option(BUILD_UTILS "Utility programs from the 'util' directory will be built." OFF)
option(BUILD_EXAMPLES "Example programs from the 'examples' directory will be built." OFF)

set(CMAKE_C_COMPILER gcc-5)
set(CMAKE_C_FLAGS "-Wall -fopenmp -Wno-unused-result -O2")
set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

list(APPEND LINK_LIBS m)

if(BUILD_TESTS)
    add_executable(max_pooling_test ${PROJECT_SOURCE_DIR}/test/max_pooling_test.c)
    target_link_libraries(max_pooling_test ${LINK_LIBS})

    add_executable(convolution_benchmark ${PROJECT_SOURCE_DIR}/test/convolution_benchmark.c)
    target_link_libraries(convolution_benchmark ${LINK_LIBS} jpeg)

    add_executable(activation_function_benchmark ${PROJECT_SOURCE_DIR}/test/activation_function_benchmark.c)
    target_link_libraries(activation_function_benchmark ${LINK_LIBS} jpeg)
endif()

if(BUILD_UTILS)
    add_executable(compare_float_files ${PROJECT_SOURCE_DIR}/util/compare_float_files.c)

    add_executable(display_pgm_cmd ${PROJECT_SOURCE_DIR}/util/display_pgm_cmd.c)
    target_link_libraries(display_pgm_cmd ${LINK_LIBS})

    add_executable(jpeg_to_rgb_pgms ${PROJECT_SOURCE_DIR}/util/jpeg_to_rgb_pgms.c)
	target_link_libraries(jpeg_to_rgb_pgms ${LINK_LIBS} jpeg)
endif()

if(BUILD_EXAMPLES)
	#add_executable(lenet_tanh ${PROJECT_SOURCE_DIR}/examples/lenet_tanh.c)
	#target_link_libraries(lenet_tanh ${LINK_LIBS})

    add_executable(lenet_caffe ${PROJECT_SOURCE_DIR}/examples/lenet_caffe.c)
    target_link_libraries(lenet_caffe ${LINK_LIBS})

	add_executable(es_alexnet ${PROJECT_SOURCE_DIR}/examples/es_alexnet.c)
    target_link_libraries(es_alexnet ${LINK_LIBS} jpeg)
endif()

# removes pgm and float files
add_custom_target(remove_output_data
    COMMAND rm ${CMAKE_BINARY_DIR}/*.pgm
    COMMAND rm ${CMAKE_BINARY_DIR}/*.float
)


