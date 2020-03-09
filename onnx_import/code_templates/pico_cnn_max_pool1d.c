{% if padding_needed %}
const uint16_t {{identifier}}_padding[2] = { {{padding.0}}, {{padding.1}} };
for (uint32_t i = 0; i < {{num_input_channels}}; i++) {
    max_pooling1d_naive_padded({{input_buffer.name}}[i],
                               {{input_width}}, {{output_buffer.name}}[i],
                               {{kernel_size}}, {{kernel_stride}}, {{identifier}}_padding);
}
{% else %}
for (uint32_t i = 0; i < {{num_input_channels}}; i++) {
    max_pooling1d_naive({{input_buffer.name}}[i],
                        {{input_width}}, {{output_buffer.name}}[i],
                        {{kernel_size}}, {{kernel_stride}});
}
{% endif %}