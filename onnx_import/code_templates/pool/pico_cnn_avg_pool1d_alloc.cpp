{% if padding_needed %}
    uint32_t {{identifier}}_kernel_shape[1] = { {{kernel_shape.0}} };
    uint32_t {{identifier}}_stride[1] = { {{stride.0}} };
    uint32_t {{identifier}}_padding[2] = { {{padding.0}}, {{padding.1}} };

    {{identifier}}_layer = new pico_cnn::naive::AveragePooling("{{name}}", 0, pico_cnn::op_type::AveragePool,
                                                  {{identifier}}_kernel_shape, {{identifier}}_stride, {{identifier}}_padding, {{count_include_pad}});
{% else %}
    uint32_t {{identifier}}_kernel_shape[1] = { {{kernel_shape.0}} };
    uint32_t {{identifier}}_stride[1] = { {{stride.0}} };

    {{identifier}}_layer = new pico_cnn::naive::AveragePooling("{{name}}", 0, pico_cnn::op_type::AveragePool,
                                                  {{identifier}}_kernel_shape, {{identifier}}_stride, nullptr, {{count_include_pad}});
{% endif %}

