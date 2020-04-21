{% if num_dims == 4 %}
{{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_output_channels}}, {{num_input_channels}}, {{kernel_height}}, {{kernel_width}});
{{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
{% elif num_dims == 3 %}
{{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_output_channels}}, {{num_input_channels}}, {{kernel_width}});
{{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
{% elif num_dims == 2 %}
{{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_output_channels}}, {{num_input_channels}});
{{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
{% elif num_dims == 1 %}
{{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_output_channels}});
{{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
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