{% if one_dimensional %}
{{buffer_name}} = (fp_t*) malloc({{num_kernels}} * sizeof(fp_t));
{% else %}
{{buffer_name}} = (fp_t**) malloc({{num_kernels}} * sizeof(fp_t*));

for(uint32_t kernel = 0; kernel < {{num_kernels}}; kernel++){
    {{buffer_name}}[kernel] = (fp_t*) malloc({{kernel_height}}*{{kernel_width}} * sizeof(fp_t));
}
{% endif %}
{%if pos >= 0 %}
{%if buffer_type == "kernel" %}
kernels[{{pos_kernel}}] = {{buffer_name}};
{% elif buffer_type == "bias" %}
biases[{{pos_bias}}] = {{buffer_name}};
{% elif buffer_type == "kernel2" %}
kernels[{{pos_kernel}}] = &{{buffer_name}};
{% endif %}
{% endif %}