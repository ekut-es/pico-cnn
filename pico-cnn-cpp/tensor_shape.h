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

            TensorShape(const TensorShape &other);

            TensorShape(uint32_t x1);

            TensorShape(uint32_t x1, uint32_t x2);

            TensorShape(uint32_t x1, uint32_t x2, uint32_t x3);

            TensorShape(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4);

            TensorShape(size_t num_dimensions);

            ~TensorShape();

            size_t num_dimensions();

            void set_num_dimensions(size_t num_dims);

            uint32_t *shape();

            void set_shape_idx(size_t idx, uint32_t value);

            uint32_t total_size();

            uint32_t operator[](size_t dim) const;

            uint32_t &operator[](size_t dim);

            friend std::ostream &operator<<(std::ostream &out, TensorShape const &tensor_shape);

        private:
            size_t num_dimensions_;
            uint32_t *shape_;
        };
    }
}

#endif //PICO_CNN_TENSOR_SHAPE_H
