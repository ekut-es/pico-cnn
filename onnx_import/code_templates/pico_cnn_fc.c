{% if bias_buffer %}
    auto *{{identifier}}_layer = new pico_cnn::naive::FullyConnected("{{name}}", 0, pico_cnn::op_type::Gemm, {{weight_buffer.name}}, {{bias_buffer.name}});
{% else %}
    auto *{{identifier}}_layer = new pico_cnn::naive::FullyConnected("{{name}}", 0, pico_cnn::op_type::Gemm, {{weight_buffer.name}}, nullptr);
{% endif %}
    {{identifier}}_layer->run({{input_buffer.name}}, {{output_buffer.name}});
    delete {{identifier}}_layer;

