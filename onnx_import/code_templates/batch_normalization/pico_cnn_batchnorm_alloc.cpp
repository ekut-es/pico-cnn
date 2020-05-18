    {{identifier}}_layer = new pico_cnn::naive::BatchNormalization("{{name}}", 0, pico_cnn::op_type::BatchNormalization,
                                                                   {{gamma_buffer.name}}, {{bias_buffer.name}},
                                                                   {{mean_buffer.name}}, {{variance_buffer.name}}, {{eps}});

