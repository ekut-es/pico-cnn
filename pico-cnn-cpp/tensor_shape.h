/**
 * @brief pico_cnn::naive::TensorShape class encapsulating all shape related information of a pico_cnn::naive::Tensor
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_TENSOR_SHAPE_H
#define PICO_CNN_TENSOR_SHAPE_H

#include <cstdint>
#include <cstring>
#include <iostream>

#include "parameters.h"

namespace pico_cnn {
    namespace naive {
        class TensorShape {
        public:
            TensorShape();

            // TODO: Check if copy-constructor is possible
            //TensorShape(const TensorShape &other);

            TensorShape(uint32_t x1);

            TensorShape(uint32_t x1, uint32_t x2);

            TensorShape(uint32_t x1, uint32_t x2, uint32_t x3);

            TensorShape(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4);

            ~TensorShape();

            size_t num_dimensions() const;

            void set_num_dimensions(size_t num_dims);

            uint32_t *shape() const;

            void set_shape_idx(size_t idx, uint32_t value);

            uint32_t total_num_elements() const;

            void freeze_shape();

            uint32_t operator[](size_t dim) const;
            uint32_t &operator[](size_t dim);

            uint32_t num_batches() const;
            uint32_t num_channels() const;
            uint32_t height() const;
            uint32_t width() const;

            TensorShape *expand_with_padding(uint32_t *padding);

            bool operator ==(const TensorShape &other) const {
                if(this->num_dimensions_ != other.num_dimensions_) {
                    return false;
                } else {
                    for(uint32_t i = 0; i < num_dimensions_; i++) {
                        if(shape_[i] != other.shape_[i])
                            return false;
                    }
                    return true;
                }
            }
            bool operator !=(const TensorShape &other) const {
                return !this->operator==(other);
            }

            friend std::ostream &operator<<(std::ostream &out, TensorShape const &tensor_shape);

        private:
            size_t num_dimensions_;
            bool modifiable;
            uint32_t *shape_;
        };
    }
}

#endif //PICO_CNN_TENSOR_SHAPE_H
