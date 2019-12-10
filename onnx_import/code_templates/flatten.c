{% if no_change == 0 %}
for(int i = 0; i < {{num_input_channels}}; i++){
    memcpy(&{{output_buffer.name}}[i*{{input_height}}*{{input_width}}],
           {{input_buffer.name}}[i],
           {{input_height}}*{{input_width}}*sizeof(fp_t));
}
{% elif no_change == 1 %}
for(int i = 0; i < {{num_input_channels}}; i++){
    memcpy(&{{output_buffer.name}}[i*{{input_height}}*{{input_width}}],
           {{input_buffer.name}},
           {{input_height}}*{{input_width}}*sizeof(fp_t));
}
{% endif %}