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

#include <array>

#include "parameters.h"
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

            Tensor(uint32_t x0);
            Tensor(uint32_t x0, uint32_t x1);
            Tensor(uint32_t x0, uint32_t x1, uint32_t x2);
            Tensor(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3);

            ~Tensor();

            inline fp_t &access(uint32_t x0) const {
                return data_[x0];
            }

            inline fp_t &access(uint32_t x0, uint32_t x1,
                                uint32_t width) const {
                return data_[(x0*width) + (x1)];
            }

            inline fp_t &access(uint32_t x0, uint32_t x1, uint32_t x2,
                                uint32_t num_channels, uint32_t width) const {
                return data_[(x0*num_channels*width) + (x1*width) + (x2)];
            }

            inline fp_t &access(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
                                uint32_t num_channels, uint32_t height, uint32_t width) const {
                return data_[(x0*num_channels*height*width) + (x1*height*width) + (x2*width) + x3];
            }

            inline fp_t &access_blob(uint32_t x) const {
                return data_[x];
            }

            fp_t *get_ptr_to_channel(uint32_t x0, uint32_t x1) const;

            uint32_t size_bytes() const;

            uint32_t num_elements() const;

            uint32_t num_dimensions() const;
            uint32_t num_batches() const;
            uint32_t num_channels() const;
            uint32_t height() const;
            uint32_t width() const;

            Tensor *expand_with_padding(uint32_t *padding, fp_t initializer = 0.0) const;
            Tensor *copy_with_padding_into(Tensor *dest, uint32_t *padding, fp_t initializer = 0.0) const;

            void copy_data_into(Tensor *dest) const;

            void concatenate_from(uint32_t num_inputs, Tensor **inputs, uint32_t dimension) const;

            bool add_tensor(Tensor *other) const;
            bool add_channel(Tensor *other, uint32_t batch, uint32_t channel);

            bool operator ==(const Tensor &other) const {
                for (uint32_t i = 0; i < num_dimensions_; i++) {
                    if(shape_[i] != other.shape_[i]) {
                        return false;
                    }
                }
                for(uint32_t i = 0; i < num_elements_; i++) {
                    if(!fp_t_eq(this->data_[i], other.data_[i])) {
                        return false;
                    }
                }
                return true;
            }

            friend std::ostream &operator<<(std::ostream &out, Tensor const &tensor);

            //private:
            const uint32_t num_dimensions_;
            uint32_t *shape_;
            fp_t *data_;
            uint32_t num_elements_;
        };
    }
}

#endif //PICO_CNN_TENSOR_H
