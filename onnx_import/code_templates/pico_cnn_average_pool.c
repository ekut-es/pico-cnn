for (int i = 0; i < {{input_channels}}; i++) {
  average_pooling1d_naive(&({{input_buffer.start_ptr}}[i * {{input_width}}]),
                          {{input_width}},
                          &({{output_buffer.start_ptr}}[i * {{output_width}}]),
                          {{kernel_size}},
                          {% if bias_buffer  %}
                          {{bias_buffer.start_ptr}}[i]
                          {% else %}
                          0
                          {% endif %}
    );
}
