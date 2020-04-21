from ir import *

from jinja2 import Environment, FileSystemLoader
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "code_templates")

template_env = Environment(loader=FileSystemLoader(template_dir))

__author__ = "Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


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
    def create(cls, buffer, pos=-1, pos_kernel=-1, pos_bias=-1):
        pass


class KernelAllocationCode(BaseCode):
    """
    Class implementing generation of memory allocation code for kernels and biases.
    """
    name = "KernelAllocation"
    template_file = "kernel_allocation.c"

    @classmethod
    def create(cls, buffer, pos=-1, pos_kernel=-1, pos_bias=-1):
        """
        Derive necessary information from the shapes of the inputs and pass them to the code template.
        :param buffer: Buffer object containing different information about the kernel/bias input.
        :param pos: Position of the kernel/bias-array when moving through the CNN.
        Needed for reading weights from binary weights file.
        :param pos_kernel: Position of the kernel-array when moving through the CNN.
        Needed for reading weights from binary weights file.
        :param pos_bias: Position of the bias-array when moving through the CNN.
        Needed for reading weights from binary weights file.
        :return: KernelAllocationCode object
        """
        operation = cls(buffer)
        buffer_shape = buffer.shape

        # print("Kernel shape: {}".format(str(buffer_shape)))
        if len(buffer_shape) == 4:
            num_dims = 4
            num_output_channels = buffer_shape[0]
            num_input_channels = buffer_shape[1]
            kernel_height = buffer_shape[2]
            kernel_width = buffer_shape[3]
            buffer_type = "kernel"
        elif len(buffer_shape) == 1:
            num_dims = 1
            operation.attributes['one_dimensional'] = 1
            num_output_channels = buffer_shape[0]
            num_input_channels = 0
            kernel_height = kernel_width = 0
            buffer_type = "bias"
        elif len(buffer_shape) == 2:
            num_dims = 2
            operation.attributes['one_dimensional'] = 1
            num_output_channels = buffer_shape[0]
            num_input_channels = buffer_shape[1]
            kernel_height = kernel_width = 0
            buffer_type = "kernel2"
        elif len(buffer_shape) == 3:
            num_dims = 3
            operation.attributes['one_dimensional'] = 0
            num_output_channels = buffer_shape[0]
            num_input_channels = buffer_shape[1]
            kernel_height = 1
            kernel_width = buffer_shape[2]
            buffer_type = 'kernel'
        else:
            print("ERROR: Unknown kernel shape: {}, Buffer: {}".format(str(buffer_shape), buffer.name))
            num_dims = 0
            num_output_channels = 0
            num_input_channels = 0
            kernel_width = kernel_height = 0
            exit(1)

        operation.attributes['buffer_name'] = buffer.name
        operation.attributes['num_dims'] = num_dims
        operation.attributes['num_output_channels'] = num_output_channels
        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['kernel_height'] = kernel_height
        operation.attributes['kernel_width'] = kernel_width
        operation.attributes['data_type'] = buffer.dt_string
        operation.attributes['pos'] = pos
        operation.attributes['pos_kernel'] = pos_kernel
        operation.attributes['pos_bias'] = pos_bias
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
    def create(cls, buffer, pos=-1, pos_kernel=-1, pos_bias=-1):
        """
        Derive necessary information from the shapes of the inputs and pass them to the code template.
        :param buffer: Buffer object containing different information about the output buffer.
        :param pos: Not needed for generation of output buffer allocation code.
        :param pos_kernel: Not needed for generation of output buffer allocation code.
        :param pos_bias: Not needed for generation of output buffer allocation code.
        :return: OutputAllocation object
        """
        operation = cls(buffer)
        buffer_shape = buffer.shape

        if len(buffer_shape) == 4:
            num_dims = 4
            num_batches = buffer_shape[0]
            num_channels = buffer_shape[1]
            height = buffer_shape[2]
            width = buffer_shape[3]
        elif len(buffer_shape) == 3:
            num_dims = 3
            num_batches = buffer_shape[0]
            num_channels = buffer_shape[1]
            height = 1
            width = buffer_shape[2]
        elif len(buffer_shape) == 2:
            num_dims = 2
            num_batches = buffer_shape[0]
            num_channels = buffer_shape[1]
            height = 0
            width = 0
        elif len(buffer_shape) == 1:
            num_dims = 1
            num_batches = buffer_shape[0]
            num_channels = 0
            height = 0
            width = 0
        else:
            print("ERROR: Unknown output shape: {}, Buffer: {}".format(str(buffer_shape), buffer.name))
            num_dims = 0
            num_batches = 0
            num_channels = 0
            height = 0
            width = 0
            exit(1)

        # # print("Output buffer shape: {}".format(str(buffer_shape)))
        # if buffer.buffer_depth == 2:
        #     num_outputs = buffer_shape[0] * buffer_shape[1]
        #     if len(buffer_shape) == 4:
        #         output_height = buffer_shape[2]
        #         output_width = buffer_shape[3]
        #     elif len(buffer_shape) == 3:
        #         output_height = 1
        #         output_width = buffer_shape[2]
        #     else:
        #         print("ERROR: Unsupported output buffer shape: {}".format(buffer_shape))
        #         exit(1)
        # elif buffer.buffer_depth == 1:
        #     operation.attributes['one_dimensional'] = 1
        #     if len(buffer_shape) == 2:
        #         num_outputs = buffer_shape[0]*buffer_shape[1]
        #     elif len(buffer_shape) == 1:
        #         num_outputs = buffer_shape[0]
        #     else:
        #         print("ERROR: Unsupported buffer_shape {} for buffer_depth == 1".format(buffer_shape))
        #         exit(1)
        #
        #     output_width = output_height = 0
        # elif buffer.buffer_depth == 3:
        #     num_outputs = buffer_shape[0] * buffer_shape[1]
        #     output_width = buffer_shape[2]
        #     output_height = 1
        # else:
        #     print("ERROR: Unknown output shape: {}, Buffer: {}".format(str(buffer_shape), buffer.name))
        #     num_outputs = output_width = output_height = 0
        #     exit(1)

        operation.attributes['buffer_name'] = buffer.name
        operation.attributes['num_dims'] = num_dims
        operation.attributes['num_batches'] = num_batches
        operation.attributes['num_channels'] = num_channels
        operation.attributes['height'] = height
        operation.attributes['width'] = width
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
    def create(cls, buffer, pos=-1, pos_kernel=-1, pos_bias=-1):
        """
        Generate code that frees the allocated memory buf the specified buffer.
        :param buffer: Buffer object for which memory has to be freed again.
        :param pos: Not needed for clean-up code.
        :return: BufferCleanup object
        """
        operation = cls(buffer)

        operation.attributes["buffer_name"] = buffer.name

        return operation


CodeRegistry.register(BufferCleanup)
