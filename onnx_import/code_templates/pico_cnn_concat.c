    {{input_declaration}}

    {{input_shape_code}}

    concatenate_naive({{inputs}}, {{input_shape}}, {{dimension}},
                      {{num_inputs}}, {{output_buffer.name}});

    {{cleanup_input}}