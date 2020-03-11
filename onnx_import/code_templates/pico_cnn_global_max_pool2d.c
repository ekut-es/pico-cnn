for (uint32_t i = 0; i < {{num_input_channels}}; i++) {
    global_max_pooling2d_naive({{input_buffer.name}}[i],
                               {{input_height}},
                               {{input_width}},
                               {{output_buffer.name}}[i]);
}