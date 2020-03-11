{% if one_dimensional %}
{{buffer_name}} = (fp_t*) malloc({{num_outputs}} * sizeof(fp_t));
{% else %}
{{buffer_name}} = (fp_t**) malloc({{num_outputs}} * sizeof(fp_t*));

for(uint32_t output = 0; output < {{num_outputs}}; output++){
    {{buffer_name}}[output] = (fp_t*) malloc({{output_height}}*{{output_width}} * sizeof(fp_t));
}
{% endif %}