#include "tensor.h"

namespace pico_cnn {

    Tensor::Tensor() {
        shape_ = TensorShape();
        data_ = nullptr;
    }

    Tensor::Tensor(pico_cnn::Tensor &other) {
        shape_ = TensorShape(other.shape_);
        data_ = new fp_t[shape_.total_size()]();
        std::memcpy(data_, other.data_, shape_.total_size());
    }

    Tensor::Tensor(TensorShape &shape) {
        shape_ = shape;
        data_ = new fp_t[shape_.total_size()]();
    }

    Tensor::~Tensor() {
        delete[](data_);
    }

    TensorShape &Tensor::shape() {
        return shape_;
    }

//    std::ostream& operator<< (std::ostream &out, Tensor const& tensor) {
//        out << "shape: " << tensor.shape_ << ", ";
//
//        return out;
//    }

}