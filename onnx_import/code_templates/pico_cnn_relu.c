
    auto *{{identifier}}_layer = new pico_cnn::naive::ReLU("{{name}}", 0, pico_cnn::op_type::ReLU);
    {{identifier}}_layer->run({{input_buffer.name}}, {{output_buffer.name}});
    delete {{identifier}}_layer;

