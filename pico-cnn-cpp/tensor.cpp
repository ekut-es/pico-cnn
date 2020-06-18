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

        uint32_t Tensor::num_dimensions() const {
            return shape_->num_dimensions();
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

        void Tensor::copy_data_into(Tensor *dest) {
            if(this->num_elements() == dest->num_elements()) {
                std::memcpy(dest->data_, this->data_, size_bytes());
            } else {
                PRINT_ERROR_AND_DIE("Attempted to copy Tensors of unequal number of elements.")
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

        bool Tensor::add_channel(Tensor *other, uint32_t batch, uint32_t channel) {
            uint32_t height = this->height();
            uint32_t width = this->width();
            uint32_t other_height = other->height();
            uint32_t other_width = other->width();

            if (height == other_height && width == other_width) {

                for (uint32_t i = 0; i < height; i++) {
                    for (uint32_t j = 0; j < width; j++) {
                        this->access(batch, channel, i, j) = this->access(batch, channel, i, j) + other->access(batch, channel, i, j);
                    }
                }

                return true;

            } else {
                PRINT_ERROR("Channels of different height and width cannot be added.");
                return false;
            }
        }

        fp_t &Tensor::access(uint32_t x, ...) {
            va_list args;
            va_start(args, x);

            uint32_t num_dims = shape_->num_dimensions();
            uint32_t indexes[num_dims];

            indexes[0] = x;

            for(size_t i = 1; i < num_dims; i++) {
                indexes[i] = va_arg(args, uint32_t);
            }

            if(num_dims == 1) {

                return data_[x];

            } else if (num_dims == 2) {

                uint32_t *shape = shape_->shape();
                return data_[(indexes[0]*shape[1]) + (indexes[1])];

            } else if (num_dims == 3) {

                uint32_t *shape = shape_->shape();
                return data_[(indexes[0]*shape[1]*shape[2]) + (indexes[1]*shape[2]) + (indexes[2])];

            } else if (num_dims == 4) {

                uint32_t *shape = shape_->shape();
                return data_[(indexes[0]*shape[1]*shape[2]*shape[3]) + (indexes[1]*shape[2]*shape[3]) + (indexes[2]*shape[3]) + indexes[3]];

            } else {

                uint32_t offset = 0;
                uint32_t *shape = shape_->shape();
                for (size_t i = 0; i < num_dims; i++) {
                    offset += product(reinterpret_cast<int32_t *>(shape), i+1, num_dims) * indexes[i];
                }
                return *(data_+offset);

            }
        }

        fp_t &Tensor::access_blob(uint32_t x) {
            return data_[x];
        }

        // TODO: Refactor get_ptr_to_channel()
        fp_t *Tensor::get_ptr_to_channel(uint32_t x, ...) const {
            va_list args;
            va_start(args, x);

            uint32_t num_dims = shape_->num_dimensions();
            uint32_t num_idx = 1;
            if (num_dims == 4) {
                num_idx = 2;
            } else if (num_dims == 3) {
                num_idx = 2;
            } else if (num_dims == 2) {
                num_idx = 1;
            } else if (num_dims == 1) {
                num_idx = 1;
            }
            uint32_t indexes[num_idx];

            indexes[0] = x;

            for(size_t i = 1; i < num_idx; i++) {
                indexes[i] = va_arg(args, uint32_t);
            }

            if(num_dims == 1) {

                return data_;

            } else if (num_dims == 2) {

                //PRINT_WARNING("Assuming that we deal with 2D data.")
                uint32_t *shape = shape_->shape();
                return data_;

            } else if (num_dims == 3) {
                // TODO: Check if this works, assuming shape = {num_batches, num_channels, width} while height = 1
                uint32_t *shape = shape_->shape();
                return data_ + (indexes[0]*shape[1]*shape[2]) + (indexes[1]*shape[2]);

            } else if (num_dims == 4) {

                uint32_t *shape = shape_->shape();
                return data_ + (indexes[0]*shape[1]*shape[2]*shape[3]) + (indexes[1]*shape[2]*shape[3]);

            } else {
                PRINT_ERROR_AND_DIE("Not implemented for shape: " << this->shape_);
                return nullptr;
            }
        }

        std::ostream &operator<<(std::ostream &out, Tensor const &tensor) {
            out << "shape: " << *(tensor.shape()) << std::endl;
            out << "data:" << std::endl;
            if (tensor.num_dimensions() == 4 || tensor.num_dimensions() == 3 ) {

                uint32_t num_batches = tensor.num_batches();
                uint32_t num_channels = tensor.num_channels();
                uint32_t height = tensor.height();
                uint32_t width = tensor.width();

                fp_t *channel_ptr;

                for (uint32_t batch = 0; batch < num_batches; batch++) {
                    out << "[";
                    for (uint32_t channel = 0; channel < num_channels; channel++) {
                        if (channel > 0) {
                            out << std::string(1, ' ' ) << "[";
                        } else {
                            out << "[";
                        }

                        channel_ptr = tensor.get_ptr_to_channel(batch, channel);
                        for (uint32_t row = 0; row < height; row++) {

                            if (row > 0) {
                                out << std::string(2, ' ') << "[";
                            } else {
                                out << "[";
                            }
                            for (uint32_t col = 0; col < width; col++) {

                                out << channel_ptr[row*width + col];
                                if (col != width-1)
                                    out << ", ";
                            }

                            if (row == height-1)
                                out << "]";
                            else
                                out << "]" << std::endl;
                        }

                        if (channel == num_channels-1)
                            out << "]" ;
                        else
                            out << "]" << std::endl;
                    }
                    if (batch == num_batches-1)
                        out << "]" ;
                    else
                        out << "]" << std::endl;
                }

            } else if (tensor.num_dimensions() == 2) {

                uint32_t height = tensor.height();
                uint32_t width = tensor.width();

                fp_t *channel_ptr;

                out << "[";

                channel_ptr = tensor.get_ptr_to_channel(0, 0);
                for (uint32_t row = 0; row < height; row++) {

                    if (row > 0) {
                        out << std::string(1, ' ') << "[";
                    } else {
                        out << "[";
                    }
                    for (uint32_t col = 0; col < width; col++) {

                        out << channel_ptr[row*width + col];
                        if (col != width-1)
                            out << ", ";
                    }

                    if (row == height-1)
                        out << "]";
                    else
                        out << "]" << std::endl;
                }

                out << "]";

            } else {
                out << "[";
                for (uint32_t i = 0; i < tensor.num_elements(); i++) {
                    if (i == tensor.num_elements() - 1) {
                        out << tensor.data_[i] << "]";
                    } else {
                        out << tensor.data_[i] << ", ";
                    }
                }
            }
            return out;
        }

        Tensor *Tensor::expand_with_padding(uint32_t *padding, fp_t initializer) {

            TensorShape *extended_shape = this->shape_->expand_with_padding(padding);

            if(extended_shape) {

                Tensor *extended_tensor = new Tensor(extended_shape);

                return this->copy_with_padding_into(extended_tensor, padding, initializer);

            } else {
                PRINT_ERROR_AND_DIE("TensorShape expansion failed.");
                return nullptr;
            }
        }

        Tensor *Tensor::copy_with_padding_into(Tensor *dest, uint32_t *padding, fp_t initializer) {

            uint32_t width_padded = dest->width();
            uint32_t height = this->height();
            uint32_t width = this->width();

            TensorShape* dest_shape = dest->shape();

            if(initializer != 0.0) {
                for (uint32_t i = 0; i < dest->num_elements(); i++) {
                    dest->access_blob(i) = initializer;
                }
            }

            if (dest_shape->num_dimensions() == 4) {

                uint32_t num_batches = dest_shape->num_batches();
                uint32_t num_channels = dest_shape->num_channels();

                fp_t *channel_ptr;
                fp_t *extended_channel_ptr;

                for (uint32_t batch = 0; batch < num_batches; batch++) {
                    for (uint32_t channel = 0; channel < num_channels; channel++) {

                        channel_ptr = this->get_ptr_to_channel(batch, channel);
                        extended_channel_ptr = dest->get_ptr_to_channel(batch, channel);

                        for (uint32_t row = 0; row < height; row++) {

                            std::memcpy((extended_channel_ptr + (row + padding[0]) * width_padded + padding[1]),
                                        channel_ptr + row * width, width * sizeof(fp_t));

                        }

                    }
                }

            } else if (dest_shape->num_dimensions() == 3) {

                uint32_t num_batches = dest_shape->num_batches();
                uint32_t num_channels = dest_shape->num_channels();

                fp_t *channel_ptr;
                fp_t *extended_channel_ptr;

                for (uint32_t batch = 0; batch < num_batches; batch++) {
                    for (uint32_t channel = 0; channel < num_channels; channel++) {

                        channel_ptr = this->get_ptr_to_channel(batch, channel);
                        extended_channel_ptr = dest->get_ptr_to_channel(batch, channel);

                        for (uint32_t row = 0; row < height; row++) {
                            std::memcpy((extended_channel_ptr + padding[0]), channel_ptr, width * sizeof(fp_t));
                        }
                    }
                }

            } else if (dest_shape->num_dimensions() == 2) {

                fp_t *channel_ptr;
                fp_t *extended_channel_ptr;


                channel_ptr = this->get_ptr_to_channel(0);
                extended_channel_ptr = dest->get_ptr_to_channel(0);

                for (uint32_t row = 0; row < height; row++) {

                    std::memcpy((extended_channel_ptr + (row + padding[0]) * width_padded + padding[1]),
                                channel_ptr + row * width, width * sizeof(fp_t));

                }

            } else if (dest_shape->num_dimensions() == 1) {
                PRINT_ERROR_AND_DIE("Not implemented for shape: " << *this->shape_);
                return nullptr;
            } else {

                PRINT_ERROR_AND_DIE("Not implemented for shape: " << *this->shape_);
                return nullptr;

            }

            return dest;
        }

        void Tensor::concatenate_from(uint32_t num_inputs, Tensor **inputs, uint32_t dimension) {

            // concatenate along channels
            if(dimension == 1) {
                uint32_t output_channel_counter = 0;

                for(uint32_t input_id = 0; input_id < num_inputs; input_id++){

                    Tensor *input = inputs[input_id];

                    uint32_t num_batches = input->num_batches();
                    uint32_t num_input_channels = input->num_channels();
                    uint32_t input_channel_size = input->height() * input->width();

                    fp_t *input_channel_ptr;
                    fp_t *output_channel_ptr;

                    for (uint32_t input_channel = 0; input_channel < num_input_channels; input_channel++) {

                        input_channel_ptr = input->get_ptr_to_channel(0, input_channel);
                        output_channel_ptr = this->get_ptr_to_channel(0, output_channel_counter+input_channel);

                        std::memcpy(output_channel_ptr,
                                    input_channel_ptr,
                                    input_channel_size * sizeof(fp_t));
                    }
                    output_channel_counter += num_input_channels;
                }

            } else {
                PRINT_ERROR_AND_DIE("ERROR: Concatenation (3-dimensional) operation not supported for dimension: " << dimension);
            }
        }

    }
}