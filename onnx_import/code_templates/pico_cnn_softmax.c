
    auto *{{identifier}}_layer = new pico_cnn::naive::Softmax("{{name}}", 0, pico_cnn::op_type::Softmax);
    {{identifier}}_layer->run({{input_buffer.name}}, {{output_buffer.name}});
    delete {{identifier}}_layer;

