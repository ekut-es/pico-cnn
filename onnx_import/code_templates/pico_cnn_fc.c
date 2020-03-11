{% if bias_buffer %}
fully_connected_naive({{input_buffer.name}}, {{input_size}},
                      {{output_buffer.name}}, {{output_size}},
                      {{weight_buffer.name}}, {{bias_buffer.name}});
{% else %}
fully_connected_naive({{input_buffer.name}}, {{input_size}},
                      {{output_buffer.name}}, {{output_size}},
                      {{weight_buffer.name}}, NULL);
{% endif %}
