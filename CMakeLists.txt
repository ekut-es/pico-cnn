cmake_minimum_required(VERSION 2.8)
project(pico_cnn)
set(CMAKE_C_FLAGS "-Wall")
set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}")

add_executable(lenet_tanh ${PROJECT_SOURCE_DIR}/examples/lenet_tanh.c)
target_link_libraries(lenet_tanh m)

add_executable(lenet_caffe ${PROJECT_SOURCE_DIR}/examples/lenet_caffe.c)
target_link_libraries(lenet_caffe m)

add_executable(compare_float_files ${PROJECT_SOURCE_DIR}/pico-cnn/util/compare_float_files.c)
