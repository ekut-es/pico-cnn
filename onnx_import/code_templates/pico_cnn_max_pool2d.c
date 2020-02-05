{% if padding_needed %}
const int {{identifier}}_padding[4] = { {{padding.0}}, {{padding.1}}, {{padding.2}}, {{padding.3}} };
for (int i = 0; i < {{num_input_channels}}; i++) {
    max_pooling2d_naive_padded({{input_buffer.name}}[i],
                                {{input_height}}, {{input_width}}, {{output_buffer.name}}[i],
                                {{kernel_size}}, {{kernel_stride}}, {{identifier}}_padding);
}
{% else %}
for (int i = 0; i < {{num_input_channels}}; i++) {
    max_pooling2d_naive({{input_buffer.name}}[i],
                      {{input_height}}, {{input_width}}, {{output_buffer.name}}[i],
                      {{kernel_size}}, {{kernel_stride}});
}
{% endif %}