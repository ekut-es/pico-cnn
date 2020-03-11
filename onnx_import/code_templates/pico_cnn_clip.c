    for (uint32_t i = 0; i < {{num_input_channels}}; i++) {
        clip_naive({{input_buffer.name}}[i],
             {{input_height}},
             {{input_width}},
             {{min}},
             {{max}},
             {{output_buffer.name}}[i]);
    }
