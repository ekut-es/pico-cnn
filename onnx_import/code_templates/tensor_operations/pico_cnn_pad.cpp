
{% if num_dims == 4 %}
    uint32_t {{identifier}}_padding[4] = { {{padding.2}}, {{padding.3}}, {{padding.6}}, {{padding.7}} };
{% elif num_dims == 3 %}
    uint32_t {{identifier}}_padding[2] = { {{padding.2}}, {{padding.5}} };
{% elif num_dims == 2 %}
    uint32_t {{identifier}}_padding[4] = { {{padding.0}}, {{padding.1}}, {{padding.2}}, {{padding.3}} };
{% endif %}
    {{input_buffer.name}}->copy_with_padding_into({{output_buffer.name}}, {{identifier}}_padding, {{initializer}});
