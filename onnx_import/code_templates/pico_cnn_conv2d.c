{% if padding_needed %}
const uint16_t {{identifier}}_padding[4] = { {{padding.0}}, {{padding.1}}, {{padding.2}}, {{padding.3}} };
for(uint32_t g = 0; g < {{num_groups}}; g++) {
    for(uint32_t i = g*{{num_output_channels}}/{{num_groups}}; i < (g+1)*{{num_output_channels}}/{{num_groups}}; i+=1){
        convolution2d_padding_naive({{input_buffer.name}}[g*{{num_input_channels}}/{{num_groups}}],
                                    {{input_height}},
                                    {{input_width}},
                                    {{output_buffer.name}}[i],
                                    {{kernel.name}}[i*{{num_input_channels}}/{{num_groups}}],
                                    {{kernel_height}},
                                    {{kernel_width}},
                                    {{stride_height}},
                                    {{stride_width}},
                                    {{identifier}}_padding,
                                    {% if bias_buffer %}
                                    {{bias_buffer.name}}[i]
                                    {% else %}
                                    0
                                    {% endif %});
        {% if num_input_channels > num_groups %}
        uint32_t cnt = 1;
        for(uint32_t j = g*{{num_input_channels}}/{{num_groups}}+1; j < (g+1)*{{num_input_channels}}/{{num_groups}}; j+=1){
            static fp_t temp_buffer[{{output_feature_size}}*{{output_feature_size}}];
            convolution2d_padding_naive({{input_buffer.name}}[j],
                                        {{input_height}},
                                        {{input_width}},
                                        temp_buffer,
                                        {{kernel.name}}[i*{{num_input_channels}}/{{num_groups}}+cnt],
                                        {{kernel_height}},
                                        {{kernel_width}},
                                        {{stride_height}},
                                        {{stride_width}},
                                        {{identifier}}_padding,
                                        0.0);

            add_channel2d_naive({{output_buffer.name}}[i],
                                temp_buffer,
                                {{output_feature_size}},
                                {{output_feature_size}});
            cnt+=1;
        }
        {% endif %}
    }
}
{% else %}
for(uint32_t g = 0; g < {{num_groups}}; g++) {
    for(uint32_t i = g*{{num_output_channels}}/{{num_groups}}; i < (g+1)*{{num_output_channels}}/{{num_groups}}; i+=1){
        convolution2d_naive({{input_buffer.name}}[g*{{num_input_channels}}/{{num_groups}}],
                            {{input_height}},
                            {{input_width}},
                            {{output_buffer.name}}[i],
                            {{kernel.name}}[i*{{num_input_channels}}/{{num_groups}}],
                            {{kernel_height}},
                            {{kernel_width}},
                            {{stride_height}},
                            {{stride_width}},
                            {% if bias_buffer %}
                            {{bias_buffer.name}}[i]
                            {% else %}
                            0
                            {% endif %});
        {% if num_input_channels > num_groups %}
        uint32_t cnt = 1;
        for(uint32_t j = g*{{num_input_channels}}/{{num_groups}}+1; j < (g+1)*{{num_input_channels}}/{{num_groups}}; j+=1){
            static fp_t temp_buffer[{{output_feature_size}}*{{output_feature_size}}];
            convolution2d_naive({{input_buffer.name}}[j],
                                {{input_height}},
                                {{input_width}},
                                temp_buffer,
                                {{kernel.name}}[i*{{num_input_channels}}/{{num_groups}}+cnt],
                                {{kernel_height}},
                                {{kernel_width}},
                                {{stride_height}},
                                {{stride_width}},
                                0.0);

            add_channel2d_naive({{output_buffer.name}}[i],
                                temp_buffer,
                                {{output_feature_size}},
                                {{output_feature_size}});
            cnt+=1;
        }
        {% endif %}
    }
}
{% endif %}