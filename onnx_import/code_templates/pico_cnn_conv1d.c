for(int i = 0; i < {{output_channels}}; i++){
  convolution1d_naive(&({{input_buffer.start_ptr}}[0]),
                      {{input_size}}, 
                      &({{output_buffer.start_ptr}}[i*{{output_feature_size}}]),
                      &({{weight_buffer.start_ptr}}[i*{{input_channels}}*{{size}}]),
                      {{size}},
                      {{stride}}, 
                      {{padding}},
                      {% if bias_buffer  %}
                      {{bias_buffer.start_ptr}}[i] / {{input_channels}}.0f,
                      {% else %}
                      0,
                      {% endif %}
                      {{dilation}});

  {% if input_channels >=  1 %}
  for(int j = 1; j < {{input_channels}}; j++){
    static fp_t temp_buffer[{{output_feature_size}}];

    convolution1d_naive(&({{input_buffer.start_ptr}}[j*{{input_size}}]), 
                        {{input_size}}, 
                        temp_buffer,
                        &({{weight_buffer.start_ptr}}[i*{{input_channels}}*{{size}}+j*{{size}}]), 
                        {{size}},
                        {{stride}}, 
                        {{padding}},
                        {% if bias_buffer  %}
                        &({{bias_buffer.start_ptr}}[i]) / {{input_channels}}.0f,
                        {% else %}
                        0,
                        {% endif %}
                        {{dilation}});

    add_image2d_naive(&({{output_buffer.start_ptr}}[i*{{output_feature_size}}]),
                      temp_buffer,
                      1,
                      {{output_feature_size}});
  }
  {% endif %}
}
