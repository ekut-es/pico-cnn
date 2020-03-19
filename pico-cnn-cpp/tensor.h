/**
 * @brief provides global parameters for pico-cnn
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
    class Tensor {
    public:
        Tensor();
        Tensor(Tensor &other);
        explicit Tensor(TensorShape &shape);

        ~Tensor();

        TensorShape &shape();

        //friend std::ostream& operator<< (std::ostream &out, Tensor const& tensor);

    private:
        TensorShape shape_;
        fp_t *data_;
        DataType data_type_;
    };
}

#endif //PICO_CNN_TENSOR_H
