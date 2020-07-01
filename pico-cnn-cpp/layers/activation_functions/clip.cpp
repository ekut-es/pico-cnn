#include "clip.h"
namespace pico_cnn {
    namespace naive {

        Clip::Clip(std::string name, uint32_t id, op_type op, const fp_t min, const fp_t max) :
        ActivationFunction(name, id, op), min(min), max(max) {

        }

        Clip::~Clip() {

        }

        void Clip::run(Tensor *input, Tensor *output) {
            this->activate(input, output);
        }

        void Clip::activate(Tensor *input, Tensor *output) {
            uint32_t num_elements = input->num_elements();
            for (uint32_t element = 0; element < num_elements; element++) {
                if(input->access_blob(element) < min)
                    output->access_blob(element) = min;
                else if(input->access_blob(element) > max)
                    output->access_blob(element) = max;
                else
                    output->access_blob(element) = input->access_blob(element);
            }
        }
    }
}