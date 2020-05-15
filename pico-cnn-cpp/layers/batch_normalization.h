//
// Created by junga on 15.05.20.
//

#ifndef PICO_CNN_BATCH_NORMALIZATION_H
#define PICO_CNN_BATCH_NORMALIZATION_H

#include "../parameters.h"
#include "../tensor.h"
#include "layer.h"

namespace pico_cnn {
    namespace naive {
        class BatchNormalization : Layer {
        public:
            BatchNormalization(std::string name, uint32_t id, op_type op, Tensor *gammas,
                               Tensor *betas, Tensor *means, Tensor *variances, fp_t epsilon);
            ~BatchNormalization();

            void run(Tensor *input, Tensor *output) override;

        private:
            void normalize(Tensor *input, Tensor *output, uint32_t channel);

            Tensor *gammas_;
            Tensor *betas_;
            Tensor *means_;
            Tensor *variances_;
            fp_t epsilon_;
        };
    }
}


#endif //PICO_CNN_BATCH_NORMALIZATION_H
