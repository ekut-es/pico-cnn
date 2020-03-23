/**
 * @brief pico_cnn::naive::Tensor class providing a uniform access to all tensor data that is used in Pico-CNN
 *
 * @author Alexander Jung (University of Tuebingen, Chair for Embedded Systems)
 */
#ifndef PICO_CNN_TENSOR_H
#define PICO_CNN_TENSOR_H

#include <cstdint>
#include <cstring>
#include <iostream>

#include "parameters.h"
#include "tensor_shape.h"

namespace pico_cnn {
    namespace naive {
        class Tensor {
        public:
            Tensor();

            Tensor(const Tensor &other);

            Tensor(TensorShape &shape);

            ~Tensor();

            TensorShape &shape();

            //friend std::ostream& operator<< (std::ostream &out, Tensor const& tensor);

        private:
            TensorShape shape_;
            fp_t *data_;
            //DataType data_type_;
        };
    }
}

#endif //PICO_CNN_TENSOR_H
