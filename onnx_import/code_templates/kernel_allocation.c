{% if num_dims == 4 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_output_channels}}, {{num_input_channels}}, {{kernel_height}}, {{kernel_width}});
{% elif num_dims == 3 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_output_channels}}, {{num_input_channels}}, {{kernel_width}});
{% elif num_dims == 2 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_output_channels}}, {{num_input_channels}});
{% elif num_dims == 1 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_output_channels}});
{% endif %}

{%if pos >= 0 %}
{%if buffer_type == "kernel" %}
    kernels[{{pos_kernel}}] = {{buffer_name}};
{% elif buffer_type == "bias" %}
    biases[{{pos_bias}}] = {{buffer_name}};
{% elif buffer_type == "kernel2" %}
    kernels[{{pos_kernel}}] = {{buffer_name}};
{% endif %}
{% endif %}
