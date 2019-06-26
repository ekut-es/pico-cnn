for(int i = 0; i < {{num_output_channels}}; i++){
    convolution2d_naive({{input_buffer.name}}[0],
                        {{input_height}},
                        {{input_width}},
                        {{output_buffer.name}}[i],
                        {{kernel.name}}[i*{{num_input_channels}}],
                        {{kernel_size}},
                        {{stride}},
                        {{padding}},
                        {% if bias_buffer %}
                        {{bias_buffer.name}}[i]
                        {% else %}
                        0
                        {% endif %});
    {% if num_input_channels > 1 %}
    for(int j = 1; j < {{num_input_channels}}; j++){
        static fp_t temp_buffer[{{output_feature_size}}];

        convolution2d_naive({{input_buffer.name}}[j],
                            {{input_height}},
                            {{input_width}},
                            temp_buffer,
                            {{kernel.name}}[i*{{num_input_channels}}+j],
                            {{kernel_size}},
                            {{stride}},
                            {{padding}},
                            0.0);

         add_channel2d_naive({{output_buffer.name}}[i],
                           temp_buffer,
                           {{output_feature_size}},
                           {{output_feature_size}});
    }
    {% endif %}
}