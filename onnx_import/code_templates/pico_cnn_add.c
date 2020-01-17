for(int i = 0; i < {{num_channels}}; i++){
    memcpy({{output_buffer.name}}[i],
           {{input_buffers[0].name}}[i],
           {{height}}*{{width}}*sizeof(fp_t));
}
{% for input_buffer in input_buffers[1:] %}
for(int i = 0; i < {{num_channels}}; i++){
    add_channel2d_naive({{output_buffer.name}}[i], {{input_buffer.name}}[i], {{height}}, {{width}});
}
{% endfor %}
