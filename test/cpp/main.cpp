#include <iostream>

#include "../../pico-cnn-cpp/pico-cnn.h"

int main() {
    std::cout << "Simple test..." << std::endl;

    pico_cnn::naive::TensorShape shape(1, 28, 28);
    std::cout << shape << std::endl;

    pico_cnn::naive::Tensor tensor(shape);
    std::cout << tensor.shape() << std::endl;

    shape.set_shape_idx(0, 3);
    std::cout << shape << std::endl;

    std::cout << tensor.shape() << std::endl;
    
}