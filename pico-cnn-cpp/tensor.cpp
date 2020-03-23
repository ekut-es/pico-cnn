#include "tensor.h"

namespace pico_cnn {
    namespace naive {

        Tensor::Tensor() {
            shape_ = TensorShape();
            data_ = nullptr;
        }

        Tensor::Tensor(const Tensor &other) {
            shape_ = TensorShape(other.shape_);
            data_ = new fp_t[shape_.total_size()]();
            std::memcpy(data_, other.data_, shape_.total_size()*sizeof(fp_t));
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
}