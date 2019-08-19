from ir import *

from jinja2 import Environment, FileSystemLoader
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "code_templates")

template_env = Environment(loader=FileSystemLoader(template_dir))


class BaseCode(object):
    """
    Base class to inherit from. Implementations see below.
    """
    def __init__(self, buffer):
        # print("Generating memory allocation code for buffer", buffer.name)
        self.buffer = buffer
        self.attributes = {}

    def generate_code(self):
        template = template_env.get_template(self.template_file)
        return template.render(**self.attributes)

    @classmethod
    def create(cls, buffer, pos=-1):
        pass


class KernelAllocationCode(BaseCode):
    """
    Class implementing generation of memory allocation code for kernels and biases.
    """
    name = "KernelAllocation"
    template_file = "kernel_allocation.c"

    @classmethod
    def create(cls, buffer, pos=-1):
        """
        Derive necessary information from the shapes of the inputs and pass them to the code template.
        :param buffer: Buffer object containing different information about the kernel/bias input.
        :param pos: Position of the kernel/bias-array when moving through the CNN. Needed for reading weights from binary weights file.
        :return: KernelAllocationCode object
        """
        operation = cls(buffer)
        buffer_shape = buffer.shape

        # print("Kernel shape: {}".format(str(buffer_shape)))
        if len(buffer_shape) == 4:
            num_kernels = buffer_shape[0] * buffer_shape[1]
            kernel_width = buffer_shape[2]
            kernel_height = buffer_shape[3]
            buffer_type = "kernel"
        elif len(buffer_shape) == 1:
            operation.attributes['one_dimensional'] = 1
            num_kernels = buffer_shape[0]
            kernel_width = kernel_height = 0
            buffer_type = "bias"
        elif len(buffer_shape) == 2:
            operation.attributes['one_dimensional'] = 1
            num_kernels = buffer_shape[0] * buffer_shape[1]
            kernel_width = kernel_height = 0
            buffer_type = "kernel2"
        else:
            print("ERROR: Unknown kernel shape: {}, Buffer: {}".format(str(buffer_shape), buffer.name))
            num_kernels = 0
            kernel_width = kernel_height = 0

        operation.attributes['buffer_name'] = buffer.name
        operation.attributes['num_kernels'] = num_kernels
        operation.attributes['kernel_height'] = kernel_height
        operation.attributes['kernel_width'] = kernel_width
        operation.attributes['data_type'] = buffer.dt_string
        operation.attributes['pos'] = pos
        operation.attributes['buffer_type'] = buffer_type

        return operation


CodeRegistry.register(KernelAllocationCode)


class OutputAllocation(BaseCode):
    """
    Class implementing generation of memory allocation code for outputs of layers (used as input for the next layer).
    """
    name = "OutputAllocation"
    template_file = "output_allocation.c"

    @classmethod
    def create(cls, buffer, pos=-1):
        """
        Derive necessary information from the shapes of the inputs and pass them to the code template.
        :param buffer: Buffer object containing different information about the output buffer.
        :param pos: Not needed for generation of output buffer allocation code.
        :return: OutputAllocation object
        """
        operation = cls(buffer)
        buffer_shape = buffer.shape

        # print("Output buffer shape: {}".format(str(buffer_shape)))
        if buffer.buffer_depth == 2:
            num_outputs = buffer_shape[0] * buffer_shape[1]
            output_width = buffer_shape[2]
            output_height = buffer_shape[3]
        elif buffer.buffer_depth == 1:
            operation.attributes['one_dimensional'] = 1
            num_outputs = buffer_shape[1]
            output_width = output_height = 0
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


class BufferCleanup(BaseCode):
    """
    Class implementing generation of code that frees all previously allocated buffers.
    """
    name = "BufferCleanup"
    template_file = "buffer_cleanup.c"

    @classmethod
    def create(cls, buffer, pos=-1):
        """
        Generate code that frees the allocated memory buf the specified buffer.
        :param buffer: Buffer object for which memory has to be freed again.
        :param pos: Not needed for clean-up code.
        :return: BufferCleanup object
        """
        operation = cls(buffer)
        buffer_shape = buffer.shape
        buffer_depth = buffer.buffer_depth

        if buffer_depth == 2:
            num_buffers = buffer_shape[0] * buffer_shape[1]
            operation.attributes["num_buffers"] = num_buffers

        operation.attributes["buffer_name"] = buffer.name
        operation.attributes["buffer_depth"] = buffer_depth

        return operation


CodeRegistry.register(BufferCleanup)
