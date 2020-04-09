#include "tensor_shape.h"
namespace pico_cnn {
    namespace naive {

        TensorShape::TensorShape() : num_dimensions_(0), shape_(nullptr), modifiable(true) {

        }

        // TODO: Check if copy-constructor is possible
//        TensorShape::TensorShape(const TensorShape &other) {
//            std::cout << "TensorShape copy constructor" << std::endl;
//            num_dimensions_ = other.num_dimensions_;
//            shape_ = new uint32_t[num_dimensions_];
//            std::memcpy(shape_, other.shape_, num_dimensions_*sizeof(uint32_t));
//        }

        TensorShape::TensorShape(uint32_t x1) : num_dimensions_(1), modifiable(true) {
            shape_ = new uint32_t[num_dimensions_];
            shape_[0] = x1;
        }

        TensorShape::TensorShape(uint32_t x1, uint32_t x2) : num_dimensions_(2), modifiable(true) {
            shape_ = new uint32_t[num_dimensions_]();
            shape_[0] = x1;
            shape_[1] = x2;
        }

        TensorShape::TensorShape(uint32_t x1, uint32_t x2, uint32_t x3) : num_dimensions_(3), modifiable(true) {
            shape_ = new uint32_t[num_dimensions_];
            shape_[0] = x1;
            shape_[1] = x2;
            shape_[2] = x3;
        }

        TensorShape::TensorShape(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4) : num_dimensions_(4), modifiable(true) {
            shape_ = new uint32_t[num_dimensions_];
            shape_[0] = x1;
            shape_[1] = x2;
            shape_[2] = x3;
            shape_[3] = x4;
        }

        TensorShape::~TensorShape() {
            delete[] shape_;
        }

        size_t TensorShape::num_dimensions() const {
            return num_dimensions_;
        }

        void TensorShape::set_num_dimensions(size_t num_dims) {
            if(modifiable) {
                if (num_dimensions_ == 0) {

                    num_dimensions_ = num_dims;
                    shape_ = new uint32_t[num_dims]();

                } else if (num_dimensions_ < num_dims) {

                    auto *new_shape = new uint32_t[num_dims]();
                    std::memcpy(new_shape, shape_, num_dimensions_ * sizeof(uint32_t));
                    delete[](shape_);

                    num_dimensions_ = num_dims;
                    shape_ = new_shape;

                } else if (num_dimensions_ == num_dims) {
                    return;
                } else {
                    if (shape_) {
                        PRINT_ERROR_AND_DIE("Reducing dimensions leads to data loss.");
                    } else {
                        num_dimensions_ = num_dims;
                        shape_ = new uint32_t[num_dims]();
                    }
                }
            } else {
                PRINT_ERROR("Trying to modify a non-modifiable TensorShape.");
            }
        }

        uint32_t *TensorShape::shape() const {
            return shape_;
        }

        void TensorShape::set_shape_idx(size_t idx, uint32_t value) {
            if(modifiable) {
                if (idx < num_dimensions_) {
                    shape_[idx] = value;
                } else {
                    PRINT_ERROR_AND_DIE("Index " << idx << " out of bounds [0," << num_dimensions_ - 1 << "].");
                }
            } else {
                PRINT_ERROR("Trying to modify a non-modifiable TensorShape.");
            }
        }

        uint32_t TensorShape::total_num_elements() const {
            if (num_dimensions_ == 0) {
                return 0;
            } else {
                uint32_t size = 1;
                for (size_t i = 0; i < num_dimensions_; i++) {
                    size *= shape_[i];
                }
                return size;
            }
        }

        void TensorShape::freeze_shape() {
            modifiable = false;
        }

        uint32_t TensorShape::operator[](size_t dim) const {
            if (dim < num_dimensions_)
                return shape_[dim];
            else
                PRINT_ERROR_AND_DIE("Dimension index out of range: %d " << dim << " >= " << num_dimensions_);
        }

        uint32_t &TensorShape::operator[](size_t dim) {
            if (dim < num_dimensions_)
                return shape_[dim];
            else
                PRINT_ERROR_AND_DIE("Dimension index out of range: %d " << dim << " >= " << num_dimensions_);
        }

        uint32_t TensorShape::num_batches() const {
            if (this->num_dimensions() == 4) {
                return shape_[0];
            } else if (this->num_dimensions() == 3) {
                return shape_[0];
            } else {
                PRINT_ERROR_AND_DIE("Cannot call num_batches() on a TensorShape with shape: " << this);
                return 0;
            }
        }

        uint32_t TensorShape::num_channels() const {
            if (this->num_dimensions() == 4) {
                return shape_[1];
            } else if (this->num_dimensions() == 3) {
                return shape_[0];
            } else if (this->num_dimensions() == 2) {
                PRINT_WARNING("Assuming 2D data with shape: " << this);
                PRINT_WARNING("Number of channels therefore assumed to be 1.");
                return 1;
            } else {
                PRINT_ERROR_AND_DIE("Cannot call num_dimensions() on a TensorShape with shape: " << this);
                return 0;
            }
        }

        uint32_t TensorShape::height() const {
            if (this->num_dimensions() == 4) {
                return shape_[2];
            } else if (this->num_dimensions() == 3) {
                return shape_[1];
            } else if (this->num_dimensions() == 2) {
                return shape_[0];
            } else {
                PRINT_ERROR_AND_DIE("Cannot call height() on a TensorShape with shape: " << this);
                return 0;
            }
        }

        uint32_t TensorShape::width() const {
            if (this->num_dimensions() == 4) {
                return shape_[3];
            } else if (this->num_dimensions() == 3) {
                return shape_[2];
            } else if (this->num_dimensions() == 2) {
                return shape_[1];
            } else {
                PRINT_ERROR_AND_DIE("Cannot call width() on a TensorShape with shape: " << this);
                return 0;
            }
        }

        std::ostream &operator<<(std::ostream &out, TensorShape const &tensor_shape) {
            out << "(";
            for (size_t i = 0; i < tensor_shape.num_dimensions_; i++) {
                if (i == tensor_shape.num_dimensions_ - 1) {
                    out << tensor_shape.shape_[i] << ")";
                } else {
                    out << tensor_shape.shape_[i] << ", ";
                }
            }
            return out;
        }

        TensorShape *TensorShape::expand_with_padding(uint32_t *padding) {

            TensorShape *expanded_shape;

            uint32_t num_dims = this->num_dimensions_;

            expanded_shape = new TensorShape();
            expanded_shape->set_num_dimensions(num_dims);

            if (num_dims == 4) {
                expanded_shape->shape_[0] = this->shape_[0];
                expanded_shape->shape_[1] = this->shape_[1];
                expanded_shape->shape_[2] = this->shape_[2] + padding[0] + padding[2];
                expanded_shape->shape_[3] = this->shape_[3] + padding[1] + padding[3];
            } else if (num_dims == 3) {
                expanded_shape->shape_[0] = this->shape_[0];
                expanded_shape->shape_[1] = this->shape_[1] + padding[0] + padding[2];
                expanded_shape->shape_[2] = this->shape_[2] + padding[1] + padding[3];
            } else if (num_dims == 2) {
                expanded_shape->shape_[0] = this->shape_[0] + padding[0] + padding[2];
                expanded_shape->shape_[1] = this->shape_[1] + padding[1] + padding[3];
            } else if (num_dims == 1) {
                expanded_shape->shape_[0] = this->shape_[0] + padding[0] + padding[1];
            } else {
                PRINT_ERROR("Extending with padding not implemented for TensorShape with number of dimensions: " << num_dims);
            }
            return expanded_shape;
        }
    }
}

