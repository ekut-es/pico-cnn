for (int i = 0; i < {{num_input_channels}}; i++) {
    batch_normalization_naive({{input_buffer.name}}[i],
                              {{input_height}},
                              {{input_width}},
                              {{output_buffer.name}}[i],
                              {{gamma_buffer.name}}[i],
                              {{bias_buffer.name}}[i],
                              {{mean_buffer.name}}[i],
                              {{variance_buffer.name}}[i],
                              {{eps}});
}
