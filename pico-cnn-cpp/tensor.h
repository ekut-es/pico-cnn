/**
 * @brief pico_cnn::naive::Tensor class providing a uniform access to all tensor data that is used in Pico-CNN
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_TENSOR_H
#define PICO_CNN_TENSOR_H

#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <iostream>
#include <zconf.h>

#include "parameters.h"
#include "tensor_shape.h"
#include "utils.h"

namespace pico_cnn {
    namespace naive {

        inline int32_t product(int32_t *array, uint32_t start, uint32_t end) {
            int32_t prod = 1;
            for(uint32_t i = start; i < end; i++) {
                prod *= array[i];
            }
            return prod;
        }

        class Tensor {
        public:
            Tensor();

            // TODO: Check if copy-constructor is possible
//            Tensor(const Tensor &other);

            Tensor(TensorShape *shape);

            ~Tensor();

            TensorShape *shape() const;

            fp_t &access(uint32_t x, ...);

            fp_t &access_blob(uint32_t x);

            uint32_t size_bytes();

            uint32_t num_elements() const;

            int32_t copy_data_into(Tensor *dest);

            bool operator ==(const Tensor &other) const {
                if(*this->shape_ == *other.shape_) {
                    uint32_t num_elements = this->shape_->total_num_elements();
                    for(uint32_t i = 0; i < num_elements; i++) {
                        if(!fp_t_eq(this->data_[i], other.data_[i]))
                            return false;
                    }
                } else {
                    return false;
                }
                return true;
            }

            friend std::ostream &operator<<(std::ostream &out, Tensor const &tensor);

        private:
            TensorShape *shape_;
            fp_t *data_;
            //DataType data_type_;
        };
    }
}

#endif //PICO_CNN_TENSOR_H
