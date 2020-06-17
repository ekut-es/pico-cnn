#include "tensor.h"

namespace pico_cnn {
    namespace naive {

        Tensor::Tensor(uint32_t x0): num_dimensions_(1) {
            shape_ = new uint32_t[num_dimensions_]();
            shape_[0] = x0;
            num_elements_ = x0;
            data_ = new fp_t [num_elements_]();
        }

        Tensor::Tensor(uint32_t x0, uint32_t x1): num_dimensions_(2) {
            shape_ = new uint32_t[num_dimensions_]();
            shape_[0] = x0;
            shape_[1] = x1;
            num_elements_ = x0*x1;
            data_ = new fp_t [num_elements_]();
        }

        Tensor::Tensor(uint32_t x0, uint32_t x1, uint32_t x2): num_dimensions_(3) {
            shape_ = new uint32_t[num_dimensions_]();
            shape_[0] = x0;
            shape_[1] = x1;
            shape_[2] = x2;
            num_elements_ = x0*x1*x2;
            data_ = new fp_t [num_elements_]();
        }

        Tensor::Tensor(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3): num_dimensions_(4) {
            shape_ = new uint32_t[num_dimensions_]();
            shape_[0] = x0;
            shape_[1] = x1;
            shape_[2] = x2;
            shape_[3] = x3;
            num_elements_ = x0*x1*x2*x3;
            data_ = new fp_t [num_elements_]();
        }

        Tensor::~Tensor() {
            delete[](data_);
            delete [] shape_;
        }

        uint32_t Tensor::size_bytes() const {
            return num_elements_ * sizeof(fp_t);
        }

        uint32_t Tensor::num_elements() const {
            return num_elements_;
        }

        uint32_t Tensor::num_dimensions() const {
            return num_dimensions_;
        }

        uint32_t Tensor::num_batches() const {
            if (num_dimensions_ == 4) {
                return shape_[0];
            } else if (num_dimensions_ == 3) {
                return shape_[0];
            } else {
                PRINT_WARNING("Assuming 1D or 2D data. Number of batches therefore assumed to be 1.")
                return 1;
            }
        }

        uint32_t Tensor::num_channels() const {
            if (num_dimensions_ == 4) {
                return shape_[1];
            } else if (num_dimensions_ == 3) {
                return shape_[1];
            } else {
                PRINT_WARNING("Assuming 1D or 2D data. Number of channels therefore assumed to be 1.")
                return 1;
            }
        }

        uint32_t Tensor::height() const {
            if (num_dimensions_ == 4) {
                return shape_[2];
            } else if (num_dimensions_ == 3) {
                return 1;
            } else if (num_dimensions_ == 2) {
                return shape_[0];
            } else {
                PRINT_ERROR_AND_DIE("Cannot call height() on a Tensor with num_dimensions not 4, 3 or 2" << this)
            }
        }

        uint32_t Tensor::width() const {
            if (num_dimensions_ == 4) {
                return shape_[3];
            } else if (num_dimensions_ == 3) {
                return shape_[2];
            } else if (num_dimensions_ == 2) {
                return shape_[1];
            } else {
                PRINT_ERROR_AND_DIE("Cannot call width() on a Tensor with num_dimensions not 4, 3 or 2" << this)
            }
        }

        void Tensor::copy_data_into(Tensor *dest) const {
            if(this->num_elements() == dest->num_elements()) {
                std::memcpy(dest->data_, this->data_, size_bytes());
            } else {
                PRINT_ERROR_AND_DIE("Attempted to copy Tensors of unequal number of elements.")
            }
        }

        bool Tensor::add_tensor(Tensor *other) const {
//            if (*(this->shape()) == *(other->shape())) {

            for (uint32_t i = 0; i < this->num_elements_; i++) {
                this->access_blob(i) = this->access_blob(i) + other->access_blob(i);
            }

            return true;

//            } else {
//                PRINT_ERROR("Tensors of different shapes cannot be added.");
//                return false;
//            }
        }

        bool Tensor::add_channel(fp_t *other, uint32_t batch, uint32_t channel) {
            uint32_t height = this->height();
            uint32_t width = this->width();

            fp_t *channel_ptr = this->get_ptr_to_channel(batch, channel);

            for (uint32_t i = 0; i < height*width; i++) {
                channel_ptr[i] = channel_ptr[i] + other[i];
            }

            return true;
        }

        fp_t *Tensor::get_ptr_to_channel(uint32_t x0, uint32_t x1) const {


            if(num_dimensions_ == 1) {

                return data_;

            } else if (num_dimensions_ == 2) {

                //PRINT_WARNING("Assuming that we deal with 2D data.")
                return data_;

            } else if (num_dimensions_ == 3) {
                // TODO: Check if this works, assuming shape = {num_batches, num_channels, width} while height = 1
                return data_ + (x0*shape_[1]*shape_[2]) + (x1*shape_[2]);

            } else if (num_dimensions_ == 4) {

                return data_ + (x0*shape_[1]*shape_[2]*shape_[3]) + (x1*shape_[2]*shape_[3]);

            } else {
                PRINT_ERROR_AND_DIE("Not implemented for num_dimensions: " << num_dimensions_)
            }
        }

        std::ostream &operator<<(std::ostream &out, Tensor const &tensor) {
            out << "shape: (";
            for (uint32_t i = 0; i < tensor.num_dimensions_; i++) {
                out << tensor.shape_[i];
                if (i != tensor.num_dimensions_-1)
                    out << ", ";
            }
            out << ")" << std::endl;
            out << "data:" << std::endl;
            if (tensor.num_dimensions_ == 4 || tensor.num_dimensions_ == 3 ) {

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

            } else if (tensor.num_dimensions_ == 2) {

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

        Tensor *Tensor::expand_with_padding(uint32_t *padding, fp_t initializer) const {

            Tensor *extended_tensor;

            if (num_dimensions_ == 4) {
                extended_tensor = new Tensor(shape_[0], shape_[1],
                                             shape_[2] + padding[0] + padding[2], shape_[3] + padding[1] + padding[3]);

            } else if (num_dimensions_ == 3) {
                extended_tensor = new Tensor(shape_[0], shape_[1],
                                             shape_[2] + padding[0] + padding[1]);
            } else if (num_dimensions_ == 2) {
                extended_tensor = new Tensor(shape_[0] + padding[0] + padding[2], shape_[1] + padding[1] + padding[3]);
            } else if (num_dimensions_ == 1) {
                extended_tensor = new Tensor(shape_[0] + padding[0] + padding[1]);
            } else {
                PRINT_ERROR_AND_DIE("Extending with padding not implemented for Tensor with number of dimensions: " << num_dimensions_)
            }
            return this->copy_with_padding_into(extended_tensor, padding, initializer);
        }

        Tensor *Tensor::copy_with_padding_into(Tensor *dest, uint32_t *padding, fp_t initializer) const {

            uint32_t width_padded = dest->width();
            uint32_t height = this->height();
            uint32_t width = this->width();

            if(initializer != 0.0) {
                for (uint32_t i = 0; i < dest->num_elements(); i++) {
                    dest->access_blob(i) = initializer;
                }
            }

            if (dest->num_dimensions_ == 4) {

                uint32_t num_batches = dest->num_batches();
                uint32_t num_channels = dest->num_channels();

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

            } else if (dest->num_dimensions_ == 3) {

                uint32_t num_batches = dest->num_batches();
                uint32_t num_channels = dest->num_channels();

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

            } else if (dest->num_dimensions_ == 2) {

                fp_t *channel_ptr;
                fp_t *extended_channel_ptr;


                channel_ptr = this->get_ptr_to_channel(0, 0);
                extended_channel_ptr = dest->get_ptr_to_channel(0, 0);

                for (uint32_t row = 0; row < height; row++) {

                    std::memcpy((extended_channel_ptr + (row + padding[0]) * width_padded + padding[1]),
                                channel_ptr + row * width, width * sizeof(fp_t));

                }

            } else if (dest->num_dimensions_ == 1) {
                PRINT_ERROR("Extending with padding not implemented for Tensor with number of dimensions: " << num_dimensions_)
            } else {
                PRINT_ERROR("Extending with padding not implemented for Tensor with number of dimensions: " << num_dimensions_)
            }

            return dest;
        }

        void Tensor::concatenate_from(uint32_t num_inputs, Tensor **inputs, uint32_t dimension) const {

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
                PRINT_ERROR_AND_DIE("ERROR: Concatenation (3-dimensional) operation not supported for dimension: " << dimension)
            }
        }

    }
}