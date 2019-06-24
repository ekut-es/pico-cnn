for (int i = 0; i < {{input_channels}}; i++) {
  clip_naive({{input_buffer.start_ptr}} +
                 i * {{input_height}} * {{input_width}},
             {{input_height}},
             {{input_width}},
             {{min}},
             {{max}},
             {{output_buffer.start_ptr}} +
                 i * {{input_height}} * {{input_width}});
}
