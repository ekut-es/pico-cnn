for(int i = 0; i < {{ input_buffer.typed_size }}; i++){
  {{ output_buffer.start_ptr }}[i] =  {{ input_buffer.start_ptr }}[i];
}
