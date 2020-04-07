#include "tensor.h"

namespace pico_cnn {
    namespace naive {

        Tensor::Tensor() {
            shape_ = nullptr;
            data_ = nullptr;
        }

        // TODO: Check if copy-constructor is possible
//        Tensor::Tensor(const Tensor &other) {
//            shape_ = TensorShape(other.shape_);
//            data_ = new fp_t[shape_.total_num_elements()]();
//            std::memcpy(data_, other.data_, shape_.total_num_elements() * sizeof(fp_t));
//        }

        Tensor::Tensor(TensorShape *shape) {
            shape->freeze_shape();
            shape_ = shape;
            uint32_t tmp = shape_->total_num_elements();
            data_ = new fp_t[shape_->total_num_elements()]();
        }

        Tensor::~Tensor() {
            delete[](data_);
        }

        TensorShape *Tensor::shape() const {
            return shape_;
        }

        uint32_t Tensor::size_bytes() {
            return shape_->total_num_elements() * sizeof(fp_t);
        }

        uint32_t Tensor::num_elements() const {
            return shape_->total_num_elements();
        }

        uint32_t Tensor::num_batches() const {
            return this->shape()->num_batches();
        }

        uint32_t Tensor::num_channels() const {
            return this->shape()->num_channels();
        }

        uint32_t Tensor::height() const {
            return this->shape()->height();
        }

        uint32_t Tensor::width() const {
            return this->shape()->width();
        }

        int32_t Tensor::copy_data_into(Tensor *dest) {
            if(*this->shape() == *dest->shape()) {
                std::memcpy(dest->data_, this->data_, size_bytes());
                return 0;
            } else {
                return -1;
            }
        }

        bool Tensor::add_tensor(Tensor *other) {
            if (*(this->shape()) == *(other->shape())) {

                for (uint32_t i = 0; i < this->num_elements(); i++) {
                    this->access_blob(i) = this->access_blob(i) + other->access_blob(i);
                }

                return true;

            } else {
                PRINT_ERROR("Tensors of different shapes cannot be added.");
                return false;
            }
        }

        fp_t &Tensor::access(uint32_t x, ...) {
            va_list args;
            va_start(args, x);

            uint32_t dims = shape_->num_dimensions();
            uint32_t indexes[dims];

            indexes[0] = x;

            for(size_t i = 1; i < dims; i++) {
                indexes[i] = va_arg(args, uint32_t);
            }

            if(dims == 1) {

                return data_[x];

            } else if (dims == 2) {

                uint32_t *shape = shape_->shape();
                return data_[(indexes[0]*shape[1]) + (indexes[1])];

            } else if (dims == 3) {

                uint32_t *shape = shape_->shape();
                return data_[(indexes[0]*shape[1]*shape[2]) + (indexes[1]*shape[2]) + (indexes[2])];

            } else if (dims == 4) {

                uint32_t *shape = shape_->shape();
                return data_[(indexes[0]*shape[1]*shape[2]*shape[3]) + (indexes[1]*shape[2]*shape[3]) + (indexes[2]*shape[3]) + indexes[3]];

            } else {

                uint32_t offset = 0;
                uint32_t *shape = shape_->shape();
                for (size_t i = 0; i < dims; i++) {
                    offset += product(reinterpret_cast<int32_t *>(shape), i+1, dims) * indexes[i];
                }
                return *(data_+offset);

            }
        }

        fp_t &Tensor::access_blob(uint32_t x) {
            return data_[x];
        }

        std::ostream &operator<<(std::ostream &out, Tensor const &tensor) {
            out << "shape: " << *(tensor.shape()) << std::endl;
            out << "data: [";
            for (uint32_t i = 0; i < tensor.num_elements(); i++) {
                if (i == tensor.num_elements() - 1) {
                    out << tensor.data_[i] << "]";
                } else {
                    out << tensor.data_[i] << ", ";
                }
            }
            return out;
        }
    }
}