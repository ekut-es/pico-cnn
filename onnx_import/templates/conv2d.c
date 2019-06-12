for(int j = 0; j < {{num_out_channels}}; j++){
    convolution2d_naive(&({{input_buffer.start_ptr}}[i]),
                        {{input_height}},
                        {{input_width}},
                        &({{output_buffer}}),
                        &({{kernel}}),
                        {{kernel_size}},
                        {{stride}},
                        {{padding}},
                        {{bias}}
                        );
}