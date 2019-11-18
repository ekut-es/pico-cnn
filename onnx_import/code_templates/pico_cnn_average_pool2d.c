{% if padding_needed %}
const int {{input_buffer.name}}_padding[4] = { {{padding.0}}, {{padding.1}}, {{padding.2}}, {{padding.3}} };
for (int i = 0; i < {{num_input_channels}}; i++) {
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
                                   {{input_buffer.name}}_padding,
                                   {{count_include_pad}});
}
{% else %}
for (int i = 0; i < {{num_input_channels}}; i++) {
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
                            {{count_include_pad}});
}
{% endif %}