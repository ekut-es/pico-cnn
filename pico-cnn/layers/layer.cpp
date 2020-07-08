#include "layer.h"

namespace pico_cnn {
    namespace naive {

        Layer::Layer(std::string name, uint32_t id, op_type op) : name_(name), id_(id), op_(op) {

        }

        Layer::~Layer() {

        }

        std::string Layer::name() {
            return name_;
        }

        uint32_t Layer::id() {
            return id_;
        }

        op_type Layer::op() {
            return op_;
        }

    }
}