{% if buffer_depth == 2 %}
for(uint32_t i = 0; i < {{num_buffers}}; i++){
    free({{buffer_name}}[i]);
}
{% endif %}
free({{buffer_name}});