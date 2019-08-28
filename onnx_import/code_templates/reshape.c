{% if no_change == 0 %}
for(int i = 0; i < {{num_input_channels}}; i++){
    memcpy(&{{output_buffer.name}}[i*{{input_width}}*{{input_height}}],
           {{input_buffer.name}}[i],
           {{input_width}}*{{input_height}}*sizeof(fp_t));
}
{% elif no_change == 1 %}
for(int i = 0; i < {{num_input_channels}}; i++){
    memcpy(&{{output_buffer.name}}[i*{{input_width}}*{{input_height}}],
           {{input_buffer.name}},
           {{input_width}}*{{input_height}}*sizeof(fp_t));
}
{% endif %}