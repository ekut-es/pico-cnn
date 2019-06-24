for (int i = 0; i < {{num_input_channels}}; i++) {
  max_pooling2d_naive({{input_buffer.start_ptr}}[i],
                      {{input_width}}, {{input_height}}, {{output_buffer}}[i],
                      {{kernel_size}}, {{kernel_stride}});
}
