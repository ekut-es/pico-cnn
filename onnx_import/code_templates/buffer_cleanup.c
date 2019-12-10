{% if buffer_depth == 2 %}
for(int i = 0; i < {{num_buffers}}; i++){
    free({{buffer_name}}[i]);
}
{% endif %}
free({{buffer_name}});