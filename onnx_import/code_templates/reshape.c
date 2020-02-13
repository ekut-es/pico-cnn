{% if no_change %}
for(int i = 0; i < {{num_input_channels}}; i++){
    memcpy(&{{output_buffer.name}}{{hotfix}}[i*{{input_height}}*{{input_width}}],
           &{{input_buffer.name}}[i],
           {{input_height}}*{{input_width}}*sizeof(fp_t));
}
{% else %}
for(int i = 0; i < {{num_input_channels}}; i++){
    memcpy(&{{output_buffer.name}}{{hotfix}}[i*{{input_height}}*{{input_width}}],
           {{input_buffer.name}}[i],
           {{input_height}}*{{input_width}}*sizeof(fp_t));
}
{% endif %}