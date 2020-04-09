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

        // TODO: Refactor get_ptr_to_channel()
        fp_t *Tensor::get_ptr_to_channel(uint32_t x, ...) {
            va_list args;
            va_start(args, x);

            uint32_t dims = shape_->num_dimensions();
            uint32_t num_idx = 1;
            if (dims == 4) {
                num_idx = 2;
            } else if (dims == 3) {
                num_idx = 1;
            } else if (dims == 2) {
                num_idx = 1;
            } else if (dims == 1) {
                num_idx = 1;
            }
            uint32_t indexes[num_idx];

            indexes[0] = x;

            for(size_t i = 1; i < num_idx; i++) {
                indexes[i] = va_arg(args, uint32_t);
            }

            if(dims == 1) {

                return data_;

            } else if (dims == 2) {

                //PRINT_WARNING("Assuming that we deal with 2D data.")
                uint32_t *shape = shape_->shape();
                return data_;

            } else if (dims == 3) {
                /* TODO: We need a solution for two cases: dims[0] is number of channels
                 * TODO: If dims[0] is number of batches, then dims[1] is number of channels and height == 1
                 */
//                uint32_t *shape = shape_->shape();
//                return data_ + (indexes[0]*shape[1]*shape[2]);
                PRINT_ERROR_AND_DIE("Not implemented for shape: " << this->shape_);
                return nullptr;

            } else if (dims == 4) {

                uint32_t *shape = shape_->shape();
                return data_ + (indexes[0]*shape[1]*shape[2]*shape[3]) + (indexes[1]*shape[2]*shape[3]);

            } else {
                PRINT_ERROR_AND_DIE("Not implemented for shape: " << this->shape_);
                return nullptr;
            }
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

        Tensor *Tensor::expand_with_padding(uint32_t *padding, fp_t initializer) {

            TensorShape *extended_shape = this->shape_->expand_with_padding(padding);

            if(extended_shape) {


                uint32_t height_padded = extended_shape->height();
                uint32_t width_padded = extended_shape->width();
                uint32_t height = this->height();
                uint32_t width = this->width();

                Tensor *extended_tensor = new Tensor(extended_shape);

                if(initializer != 0.0) {
                    for (uint32_t i = 0; i < extended_tensor->num_elements(); i++) {
                        extended_tensor->access_blob(i) = initializer;
                    }
                }

                if (extended_shape->num_dimensions() == 4) {

                    uint32_t num_batches = extended_shape->num_batches();
                    uint32_t num_channels = extended_shape->num_channels();

                    fp_t *channel_ptr;
                    fp_t *extended_channel_ptr;

                    for (uint32_t batch = 0; batch < num_batches; batch++) {
                        for (uint32_t channel = 0; channel < num_channels; channel++) {

                            channel_ptr = this->get_ptr_to_channel(batch, channel);
                            extended_channel_ptr = extended_tensor->get_ptr_to_channel(batch, channel);

                            for (uint32_t row = 0; row < height; row++) {

                                std::memcpy((extended_channel_ptr + (row + padding[0]) * width_padded + padding[1]),
                                        channel_ptr + row * width, width * sizeof(fp_t));

                            }

                        }
                    }

                } else if (extended_shape->num_dimensions() == 3) {

                    PRINT_ERROR_AND_DIE("Not implemented for shape: " << *this->shape_);
                    return nullptr;

                } else if (extended_shape->num_dimensions() == 2) {

                    fp_t *channel_ptr;
                    fp_t *extended_channel_ptr;


                    channel_ptr = this->get_ptr_to_channel(0);
                    extended_channel_ptr = extended_tensor->get_ptr_to_channel(0);

                    for (uint32_t row = 0; row < height; row++) {

                        std::memcpy((extended_channel_ptr + (row + padding[0]) * width_padded + padding[1]),
                                    channel_ptr + row * width, width * sizeof(fp_t));

                    }

                } else if (extended_shape->num_dimensions() == 1) {

                } else {

                    PRINT_ERROR_AND_DIE("Not implemented for shape: " << *this->shape_);
                    return nullptr;

                }

                return extended_tensor;

            } else {
                PRINT_ERROR_AND_DIE("TensorShape expansion failed.");
                return nullptr;
            }
        }

    }
}