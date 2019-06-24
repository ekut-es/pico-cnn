for (int i = 0; i < {{input_channels}}; i++) {
  max_pooling1d_naive(&({{input_buffer.start_ptr}}[i * {{input_width}}]),
                      {{input_width}}, &({{output_buffer}}[i * {{output_width}}]),
                      {{kernel_size}}, {{kernel_stride}});
}
