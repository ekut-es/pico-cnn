{% if dimensionality == 4 %}
    for(uint32_t i = 0; i < {{num_channels}}; i++){
        memcpy({{output_buffer.name}}[i],
               {{input_buffers[0].name}}[i],
               {{height}}*{{width}}*sizeof(fp_t));
    }
{% for input_buffer in input_buffers[1:] %}
    for(uint32_t i = 0; i < {{num_channels}}; i++){
        add_channel2d_naive({{output_buffer.name}}[i], {{input_buffer.name}}[i], {{height}}, {{width}});
    }
{% endfor %}
{% elif dimensionality == 2 %}
    for(uint32_t i = 0; i < {{num_channels}}; i++){
        memcpy({{output_buffer.name}},
               {{input_buffers[0].name}},
               {{height}}*{{width}}*sizeof(fp_t));
    }
{% for input_buffer in input_buffers[1:] %}
    for(uint32_t i = 0; i < {{num_channels}}; i++){
        add_channel2d_naive({{output_buffer.name}}, {{input_buffer.name}}, {{height}}, {{width}});
    }
{% endfor %}
{% endif %}