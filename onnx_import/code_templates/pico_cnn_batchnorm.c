for (int i = 0; i < {{input_channels}}; i++) {
  batch_normalization_naive(
      {{input_buffer.start_ptr}} + i * {{input_height}} * {{input_width}},
      {{input_height}},
      {{input_width}},
      {{mean_buffer.start_ptr}}[i],
      {{variance_buffer.start_ptr}}[i],
      {{bias_buffer.start_ptr}}[i],
      {{eps}},
      {{output_buffer.start_ptr}} +
          i * {{input_height}} * {{input_width}});
}
