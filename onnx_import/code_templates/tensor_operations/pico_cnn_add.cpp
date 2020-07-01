
    {{input_buffers[0].name}}->copy_data_into({{output_buffer.name}});
{% for input_buffer in input_buffers[1:] %}
    {{output_buffer.name}}->add_tensor({{input_buffer.name}});
{% endfor %}
