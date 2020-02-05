{% if padding_needed %}
const int {{identifier}}_padding[2] = { {{padding.0}}, {{padding.1}} };
for (int i = 0; i < {{num_input_channels}}; i++) {
    average_pooling1d_naive_padded(&({{input_buffer.name}}[i]),
                                   {{input_width}}, {{output_buffer.name}}[i],
                                   {{kernel_size}}, {{kernel_stride}},
                                   {% if bias_buffer %}
                                   {{bias_buffer.name}}[i],
                                   {% else %}
                                   0,
                                   {% endif %}
                                   {{identifier}}_padding,
                                   {{count_include_pad}});
}
{% else %}
for (int i = 0; i < {{num_input_channels}}; i++) {
    average_pooling1d_naive(&({{input_buffer.name}}[i]),
                            {{input_width}}, {{output_buffer.name}}[i],
                            {{kernel_size}}, {{kernel_stride}},
                            {% if bias_buffer %}
                            {{bias_buffer.name}}[i],
                            {% else %}
                            0,
                            {% endif %}
                            {{count_include_pad}});
}
{% endif %}