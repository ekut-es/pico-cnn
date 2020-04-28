{% if num_dims == 4 %}
    {{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_batches}}, {{num_channels}}, {{height}}, {{width}});
    {{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
{% elif num_dims == 3 %}
    {{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_batches}}, {{num_channels}}, {{width}});
    {{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
{% elif num_dims == 2 %}
    {{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_batches}}, {{num_channels}});
    {{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
{% elif num_dims == 1 %}
    {{buffer_name}}_shape = new pico_cnn::naive::TensorShape({{num_batches}});
    {{buffer_name}} = new pico_cnn::naive::Tensor({{buffer_name}}_shape);
{% endif %}
