{% if num_dims == 4 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_batches}}, {{num_channels}}, {{height}}, {{width}});
{% elif num_dims == 3 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_batches}}, {{num_channels}}, {{width}});
{% elif num_dims == 2 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_batches}}, {{num_channels}});
{% elif num_dims == 1 %}
    {{buffer_name}} = new pico_cnn::naive::Tensor({{num_batches}});
{% endif %}
