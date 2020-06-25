{% if padding_needed %}
const uint16_t {{identifier}}_padding[4] = { {{padding.0}}, {{padding.1}}, {{padding.2}}, {{padding.3}} };
for (uint32_t i = 0; i < {{num_input_channels}}; i++) {
    average_pooling2d_naive_padded({{input_buffer.name}}[i],
                                   {{input_height}},
                                   {{input_width}},
                                   {{output_buffer.name}}[i],
                                   {{kernel_size}},
                                   {{kernel_stride}},
                                   {% if bias_buffer  %}
                                   {{bias_buffer.name}}[i],
                                   {% else %}
                                   0,
                                   {% endif %}
                                   {{identifier}}_padding,
                                   {{count_include_pad}});
}
{% else %}
for (uint32_t i = 0; i < {{num_input_channels}}; i++) {
    average_pooling2d_naive({{input_buffer.name}}[i],
                            {{input_height}},
                            {{input_width}},
                            {{output_buffer.name}}[i],
                            {{kernel_size}},
                            {{kernel_stride}},
                            {% if bias_buffer  %}
                            {{bias_buffer.name}}[i],
                            {% else %}
                            0,
                            {% endif %}
                            NULL,
                            {{count_include_pad}});
}
{% endif %}