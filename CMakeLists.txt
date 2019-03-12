cmake_minimum_required(VERSION 2.8)
project(pico_cnn)

option(BUILD_TESTS "Test programs from the 'test' directory will be built." OFF)
option(BUILD_UTILS "Utility programs from the 'util' directory will be built." OFF)
option(BUILD_EXAMPLES "Example programs from the 'examples' directory will be built." OFF)

set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(LINUX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    list(APPEND LINK_LIBS m)
else()
    list(APPEND LINK_LIBS m omp)
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

    add_executable(es_alexnet_naive ${PROJECT_SOURCE_DIR}/examples/es_alexnet_naive.c)
    target_link_libraries(es_alexnet_naive ${LINK_LIBS} jpeg)

	add_executable(es_vgg_naive ${PROJECT_SOURCE_DIR}/examples/es_vgg_naive.c)
    target_link_libraries(es_vgg_naive ${LINK_LIBS} jpeg)
endif()

# removes pgm and float files
add_custom_target(remove_output_data
    COMMAND rm ${CMAKE_BINARY_DIR}/*.pgm
    COMMAND rm ${CMAKE_BINARY_DIR}/*.float
)
