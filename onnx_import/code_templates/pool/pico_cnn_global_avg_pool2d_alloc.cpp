
    {{identifier}}_layer = new pico_cnn::naive::GlobalAveragePooling("{{name}}", 0, pico_cnn::op_type::AveragePool,
                                                                     nullptr, nullptr, nullptr);
