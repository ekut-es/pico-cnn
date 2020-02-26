{% if dimensionality == 4 %}
    const int {{identifier}}_padding[4] = { {{padding.2}}, {{padding.3}}, {{padding.6}}, {{padding.7}} };
    for(int i = 0; i < {{num_input_channels}}; i++) {
        pad_2d_naive({{input_buffer.name}}[i],
                     {{input_height}},
                     {{input_width}},
                     {{output_buffer.name}}[i],
                     {{identifier}}_padding,
                     {{initializer}});
    }
{% elif dimensionality == 3 %}
    const int {{identifier}}_padding[2] = { {{padding.2}}, {{padding.5}} };
    for(int i = 0; i < {{num_input_channels}}; i++) {
        pad_1d_naive({{input_buffer.name}}[i],
                     {{input_width}},
                     {{output_buffer.name}}[i],
                     {{identifier}}_padding,
                     {{initializer}});
    }
{% elif dimensionality == 2 %}
    const int {{identifier}}_padding[4] = { {{padding.0}}, {{padding.1}}, {{padding.2}}, {{padding.3}} };
    pad_2d_naive({{input_buffer.name}},
                 {{input_height}},
                 {{input_width}},
                 {{output_buffer.name}},
                 {{identifier}}_padding,
                 {{initializer}});
{% endif %}


