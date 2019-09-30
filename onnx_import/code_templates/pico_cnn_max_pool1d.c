for (int i = 0; i < {{num_input_channels}}; i++) {
    max_pooling1d_naive({{input_buffer.name}}[i],
                        {{input_width}}, {{output_buffer.name}}[i],
                        {{kernel_size}}, {{kernel_stride}});
}
