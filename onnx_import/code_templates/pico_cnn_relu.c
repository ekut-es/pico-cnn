{% if num_input_channels > 1 %}
for (i = 0; i < {{num_input_channels}}; i++) {
    relu_naive({{input_buffer.start_ptr}} +
               i * {{input_height}} * {{input_width}},
               {{input_height}},
               {{input_width}},
               {{output_buffer.start_ptr}} +
               i * {{input_height}} * {{input_width}});
}
{% else %}
relu_naive({{input_buffer.name}}, {{input_height}}, {{input_width}}, {{output_buffer.name}});
{% endif %}