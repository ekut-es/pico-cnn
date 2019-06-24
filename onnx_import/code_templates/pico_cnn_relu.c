for (i = 0; i < {{input_channels}}; i++) {
  relu_naive({{input_buffer.start_ptr}} +
                 i * {{input_height}} * {{input_width}},
             {{input_height}},
             {{input_width}},
             {{output_buffer.start_ptr}} +
                 i * {{input_height}} * {{input_width}});
}
