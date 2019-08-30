for(int i = 0; i < {{num_input_channels}}; i++){
    memcpy(&{{output_buffer.name}}[i*{{input_width}}*{{input_height}}],
           {{input_buffer.name}}[i],
           {{input_width}}*{{input_height}}*sizeof(fp_t));
}