{% if padding_needed %}
    uint32_t {{identifier}}_padding[4] = { {{padding.0}}, {{padding.1}}, {{padding.2}}, {{padding.3}} };
    uint32_t {{identifier}}_stride[2] = { {{stride.0}}, {{stride.1}} };
    uint32_t {{identifier}}_groups = {{num_groups}};

    auto *layer = new pico_cnn::naive::Convolution("{{name}}", 0, pico_cnn::op_type::Conv,
                                                   {{kernel.name}}, {{bias_buffer.name}}, {{identifier}}_padding, {{identifier}}_stride, {{identifier}}_groups);
{% else %}
    uint32_t {{identifier}}_stride[2] = { {{stride.0}}, {{stride.1}} };
    uint32_t {{identifier}}_groups = {{num_groups}};

    auto *{{identifier}}_layer = new pico_cnn::naive::Convolution("{{name}}", 0, pico_cnn::op_type::Conv,
                                                   {{kernel.name}}, {{bias_buffer.name}}, nullptr, {{identifier}}_stride, {{identifier}}_groups);
{% endif %}
    {{identifier}}_layer->run({{input_buffer.name}}, {{output_buffer.name}});
    delete {{identifier}}_layer;

