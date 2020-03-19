#include "tensor_shape.h"
namespace pico_cnn {

    TensorShape::TensorShape(): num_dimensions_(0), shape_(nullptr) {

    }

    TensorShape::TensorShape(TensorShape &other) {
        num_dimensions_ = other.num_dimensions_;
        shape_ = new uint32_t[num_dimensions_];
        std::memcpy(shape_, other.shape_, num_dimensions_);
    }

    TensorShape::TensorShape(uint32_t x1): num_dimensions_(1) {
        shape_ = new uint32_t[num_dimensions_];
        shape_[0] = x1;
    }

    TensorShape::TensorShape(uint32_t x1, uint32_t x2): num_dimensions_(2) {
        shape_ = new uint32_t[num_dimensions_];
        shape_[0] = x1;
        shape_[1] = x2;
    }

    TensorShape::TensorShape(uint32_t x1, uint32_t x2, uint32_t x3): num_dimensions_(3) {
        shape_ = new uint32_t[num_dimensions_];
        shape_[0] = x1;
        shape_[1] = x2;
        shape_[2] = x3;
    }

    TensorShape::TensorShape(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4): num_dimensions_(4) {
        shape_ = new uint32_t[num_dimensions_];
        shape_[0] = x1;
        shape_[1] = x2;
        shape_[2] = x3;
        shape_[3] = x4;
    }

    TensorShape::TensorShape(size_t num_dimensions): num_dimensions_(num_dimensions) {
        shape_ = new uint32_t[num_dimensions]();
    }

    TensorShape::~TensorShape() {
        delete[](shape_);
    }

    size_t TensorShape::num_dimensions() {
        return num_dimensions_;
    }

    void TensorShape::set_num_dimensions(size_t num_dims) {
        if(num_dimensions_ == 0) {

            num_dimensions_ = num_dims;
            shape_ = new uint32_t[num_dims]();

        } else if(num_dimensions_ < num_dims) {

            auto *new_shape = new uint32_t[num_dims]();
            std::memcpy(new_shape, shape_, num_dimensions_);
            delete[](shape_);

            num_dimensions_ = num_dims;
            shape_ = new_shape;

        } else if(num_dimensions_ == num_dims) {
            return;
        } else {
            if(shape_) {
                PRINT_ERROR_AND_DIE("Reducing dimensions leads to data loss.");
            } else {
                num_dimensions_ = num_dims;
                shape_ = new uint32_t[num_dims]();
            }
        }
    }

    uint32_t *TensorShape::shape() {
        return shape_;
    }

    void TensorShape::set_shape_idx(size_t idx, uint32_t value) {
        if(idx < num_dimensions_) {
            shape_[idx] = value;
        } else {
            PRINT_ERROR_AND_DIE("Index " << idx << " out of bounds [0," << num_dimensions_-1 << "].");
        }
    }

    uint32_t TensorShape::total_size() {
        if(num_dimensions_ == 0) {
            return 0;
        } else {
            uint32_t size = 1;
            for (size_t i = 0; i < num_dimensions_; i++) {
                size *= shape_[i];
            }
            return size;
        }
    }

    uint32_t TensorShape::operator[](size_t dim) const {
        if(dim < num_dimensions_)
            return shape_[dim];
        else
            PRINT_ERROR_AND_DIE("Dimension index out of range: %d " << dim << " >= " << num_dimensions_);
    }

    uint32_t &TensorShape::operator[](size_t dim) {
        if(dim < num_dimensions_)
            return shape_[dim];
        else
            PRINT_ERROR_AND_DIE("Dimension index out of range: %d " << dim << " >= " << num_dimensions_);
    }

    std::ostream& operator<< (std::ostream &out, TensorShape const& tensor_shape) {
        out << "(";
        for(size_t i = 0; i < tensor_shape.num_dimensions_; i++) {
            if(i == tensor_shape.num_dimensions_-1) {
                out << tensor_shape.shape_[i] << ")";
            } else {
                out << tensor_shape.shape_[i] << ", ";
            }
        }
        return out;
    }

}

