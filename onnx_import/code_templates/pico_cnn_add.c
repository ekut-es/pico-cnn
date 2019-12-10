for(int i = 0; i < {{input_buffers[0].typed_size}}; i++){
  {{output_buffer.start_ptr}}[i] = {{input_buffers[0].start_ptr}}[i];
}


{% for input_buffer in input_buffers[1:] %}
for(int i = 0; i < {{input_buffer.typed_size}}; i++){
  {{output_buffer.start_ptr}}[i] += {{input_buffer.start_ptr}}[i];
}
{% endfor %}
