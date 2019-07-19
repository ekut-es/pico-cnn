{% if num_input_channels > 1 %}
for (int i = 0; i < {{num_input_channels}}; i++) {
    relu_naive({{input_buffer.name}}[i],
               {{input_height}},
               {{input_width}},
               {{output_buffer.name}}[i]);
}
{% else %}
relu_naive({{input_buffer.name}}, {{input_height}}, {{input_width}}, {{output_buffer.name}});
{% endif %}