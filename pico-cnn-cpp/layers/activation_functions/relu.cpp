#include "relu.h"

namespace pico_cnn {
    namespace naive {
        ReLU::ReLU(std::string name, uint32_t id, op_type op) : ActivationFunction(name, id, op) {

        }

        ReLU::~ReLU() {

        }

        void ReLU::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void ReLU::activate(Tensor *input, Tensor *output) {
            uint32_t num_elements = input->num_elements();
            for (uint32_t element = 0; element < num_elements; element++) {
                output->access_blob(element) = (input->access_blob(element) < 0.0) ? 0.0 : input->access_blob(element);
            }
        }

        LeakyReLU::LeakyReLU(std::string name, uint32_t id, op_type op, fp_t leak) :
        ActivationFunction(name, id, op), leak_(leak) {

        }

        void LeakyReLU::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void LeakyReLU::activate(Tensor *input, Tensor *output) {
            uint32_t num_elements = input->num_elements();
            for(uint32_t element = 0; element < num_elements; element++) {
                output->access_blob(element) = (input->access_blob(element) < 0.0) ?
                        (leak_*input->access_blob(element)) : input->access_blob(element);
            }
        }

        ParameterizedReLU::ParameterizedReLU(std::string name, uint32_t id, op_type op, Tensor *slope) :
        ActivationFunction(name, id, op) {
            slope_ = slope;
        }

        void ParameterizedReLU::run(Tensor *input, Tensor *output) {
            if (*(input->shape()) != *(slope_->shape())){
                PRINT_ERROR_AND_DIE("Broadcasting is not supported at the moment.")
            } else {
                this->activate(input, output);
            }
        }

        void ParameterizedReLU::activate(Tensor *input, Tensor *output) {
            uint32_t num_elements = input->num_elements();
            for(uint32_t i = 0; i < num_elements; i++) {
                output->access_blob(i) = (input->access_blob(i) < 0.0) ?
                        (slope_->access_blob(i) * input->access_blob(i)) : input->access_blob(i);
            }

        }
    }
}