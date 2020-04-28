{% if bias_buffer %}
    {{identifier}}_layer = new pico_cnn::naive::FullyConnected("{{name}}", 0, pico_cnn::op_type::Gemm, {{weight_buffer.name}}, {{bias_buffer.name}});
{% else %}
    {{identifier}}_layer = new pico_cnn::naive::FullyConnected("{{name}}", 0, pico_cnn::op_type::Gemm, {{weight_buffer.name}}, nullptr);
{% endif %}
