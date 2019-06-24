{% if one_dimensional %}
{{buffer_name}} = (fp_t*) malloc({{num_kernels}} * sizeof(fp_t));
{% else %}
{{buffer_name}} = (fp_t**) malloc({{num_kernels}} * sizeof(fp_t*));

for(int kernel = 0; kernel < {{num_kernels}}; kernel++){
    {{buffer_name}}[kernel] = (fp_t*) malloc({{kernel_height}}*{{kernel_width}} * sizeof(fp_t));
}
{% endif %}