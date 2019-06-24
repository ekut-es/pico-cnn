from ir import *

from jinja2 import Environment, FileSystemLoader
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "code_templates")

template_env = Environment(loader=FileSystemLoader(template_dir))


class BaseCode(object):
    def __init__(self, buffer):
        # print("Generating memory allocation code for buffer", buffer.name)
        self.buffer = buffer
        self.attributes = {}

    def generate_code(self):
        template = template_env.get_template(self.template_file)
        return template.render(**self.attributes)

    @classmethod
    def create(cls, buffer):
        pass


class KernelAllocationCode(BaseCode):
    name = "KernelAllocation"
    template_file = "kernel_allocation.c"

    @classmethod
    def create(cls, buffer):
        operation = cls(buffer)
        buffer_shape = buffer.shape

        # print("Kernel shape: {}".format(str(buffer_shape)))
        if len(buffer_shape) == 4:
            num_kernels = buffer_shape[0] * buffer_shape[1]
            kernel_width = buffer_shape[2]
            kernel_height = buffer_shape[3]
        elif len(buffer_shape) == 1:
            num_kernels = buffer_shape[0]
            kernel_width = 1
            kernel_height = 1
        else:
            print("ERROR: Unknown kernel shape: {}, Buffer: {}".format(str(buffer_shape), buffer.name))
            num_kernels = 0
            kernel_width = kernel_height = 0

        operation.attributes['buffer_name'] = buffer.name
        operation.attributes['num_kernels'] = num_kernels
        operation.attributes['kernel_height'] = kernel_height
        operation.attributes['kernel_width'] = kernel_width
        operation.attributes['data_type'] = buffer.dt_string

        return operation


CodeRegistry.register(KernelAllocationCode)


class OutputAllocation(BaseCode):
    name = "OutputAllocation"
    template_file = "output_allocation.c"

    @classmethod
    def create(cls, buffer):
        operation = cls(buffer)
        buffer_shape = buffer.shape

        # print("Output buffer shape: {}".format(str(buffer_shape)))
        if len(buffer_shape) == 4:
            num_outputs = buffer_shape[0] * buffer_shape[1]
            output_width = buffer_shape[2]
            output_height = buffer_shape[3]
        else:
            print("ERROR: Unknown output shape: {}, Buffer: {}".format(str(buffer_shape), buffer.name))
            num_outputs = output_width = output_height = 0

        operation.attributes['buffer_name'] = buffer.name
        operation.attributes['num_outputs'] = num_outputs
        operation.attributes['output_height'] = output_height
        operation.attributes['output_width'] = output_width
        operation.attributes['data_type'] = buffer.dt_string

        return operation


CodeRegistry.register(OutputAllocation)

