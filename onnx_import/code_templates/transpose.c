{
  {{input_def}}   = ({{input_cast}}){{input_buffer.start_ptr}};
  {{output_def}}  = ({{output_cast}}){{output_buffer.start_ptr}};

  {{ transpose_code }}

}
