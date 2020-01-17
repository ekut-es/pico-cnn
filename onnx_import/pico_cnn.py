from ir import *
from utils import reduce_mult

from jinja2 import Environment, FileSystemLoader

import numpy as np

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "code_templates")

template_env = Environment(loader=FileSystemLoader(template_dir))


class BaseLayer(object):
    """
    Base layer class to inherit from. Operator implementations see below.
    """
    def __init__(self, node, graph):
        print("Generating layer", node.name)
        self.node = node
        self.graph = graph
        self.attributes = {}

    def generate_code(self):
        template = template_env.get_template(self.template_file)
        return template.render(**self.attributes)

    @classmethod
    def create(cls, node, graph, memory_manager):
        pass


class Conv2D(BaseLayer):
    """
    2-dimensional convolution layer.
    """
    name = "PicoCNNConv2D"
    operator = "Conv"
    template_file = "pico_cnn_conv2d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        operation = cls(node, graph)

        attrs = node.attrs
        input_buffers = [memory_manager.get_buffer(graph, id) for id in node.inputs]
        output_buffers = [memory_manager.get_buffer(graph, id) for id in node.outputs]

        input_shape = input_buffers[0].shape
        num_input_channels = input_shape[1]

        if not (len(input_shape) == 4 and input_shape[3] != 1):
            print("{} is not a 2DConvolution".format(node.name))
            return None

        if input_shape[2] != input_shape[3]:
            print("WARNING: Not a squared input image ({}x{})!!!".format(input_shape[2], input_shape[3]))

        output_shape = output_buffers[0].shape
        output_size = output_shape[2]
        num_output_channels = output_shape[1]

        kernel_size = attrs["kernel_shape"][0]
        stride = attrs["strides"][0]
        #dilation = attrs["dilations"][0]

        num_groups = attrs.get("group", 1)

        # TODO: handle auto padding
        if "auto_pad" in attrs:
            print("{} auto padding is currently not supported".format(node.name))
            exit(1)

        pads = (attrs["pads"][0], attrs["pads"][1]) if len(attrs["pads"]) == 2 else (attrs["pads"][0], attrs["pads"][2])

        if pads[0] != pads[1]:
            print("PicoCNN only supports same padding in all directions")
            exit(1)

        if (kernel_size % 2) == 0:
            print("PicoCNN only supports odd kernel sizes in 2D convolution")
            exit(1)

        padding = pads[0]

        operation.attributes['input_buffer'] = input_buffers[0]
        operation.attributes['input_height'] = input_shape[2]
        operation.attributes['input_width'] = input_shape[3]
        operation.attributes['num_input_channels'] = num_input_channels

        operation.attributes['kernel'] = input_buffers[1]
        operation.attributes['kernel_size'] = kernel_size
        operation.attributes['stride'] = stride
        operation.attributes['padding'] = padding

        operation.attributes['output_buffer'] = output_buffers[0]
        operation.attributes['output_feature_size'] = output_size
        operation.attributes['num_output_channels'] = num_output_channels

        operation.attributes['num_groups'] = num_groups

        if len(input_buffers) > 2:
            operation.attributes['bias_buffer'] = input_buffers[2]

        return operation


OperationRegistry.register(Conv2D)


class Conv1D(BaseLayer):
    name = "PicoCNNConv1D"
    operator = "Conv"
    template_file = "pico_cnn_conv1d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        operation = cls(node, graph)

        attrs = node.attrs
        input_buffers = [memory_manager.get_buffer(graph, id) for id in node.inputs]
        output_buffers = [memory_manager.get_buffer(graph, id) for id in node.outputs]

        input_shape = input_buffers[0].shape

        if not (len(input_shape) == 3 or (len(input_shape) == 4 and input_shape[3] == 1)):
            print("{} is not a 1DConvolution".format(node.name))
            return None

        input_size = input_shape[2]
        num_input_channels = input_shape[1]

        output_shape = output_buffers[0].shape
        num_output_channels = output_shape[1]
        output_feature_size = output_shape[2]

        kernel_size = attrs["kernel_shape"][0]
        stride = attrs["strides"][0]
        dilation = attrs["dilations"][0]

        # TODO: handle auto padding
        if "auto_pad" in attrs:
            print("{} auto padding is currently not supported".format(node.name))
            exit(1)

        pads = (attrs["pads"][0], attrs["pads"][1]) if len(attrs["pads"]) == 2 else (attrs["pads"][0], attrs["pads"][2])

        if pads[0] != pads[1]:
            print("PicoCNN only supports same padding in all directions")
            exit(1)

        if (kernel_size % 2) == 0:
            print("PicoCNN only supports odd kernel sizes in 1D convolution")
            exit(1)

        padding = pads[0]

        operation.attributes['kernel_size'] = kernel_size
        operation.attributes['stride'] = stride
        operation.attributes['padding'] = padding
        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_size'] = input_size
        operation.attributes['num_output_channels'] = num_output_channels
        operation.attributes['padding'] = padding
        operation.attributes['input_buffer'] = input_buffers[0]
        operation.attributes['kernel'] = input_buffers[1]
        operation.attributes['output_feature_size'] = output_feature_size
        if len(input_buffers) > 2:
            operation.attributes['bias_buffer'] = input_buffers[2]

        operation.attributes['output_buffer'] = output_buffers[0]
        operation.attributes['dilation'] = dilation

        return operation


OperationRegistry.register(Conv1D)


class FullyConnected(BaseLayer):
    """
    Fully-connected layer. The corresponding operator in an onnx model is "Gemm".
    Basically this is a simple matrix x matrix multiplication.
    """
    name = "PicoCNNFullyConnected"
    operator = "Gemm"
    template_file = "pico_cnn_fc.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        attrs = node.attrs

        if 'alpha' in node.attrs:
            if node.attrs['alpha'] != 1.0:
                return None
        if 'beta' in node.attrs:
            if node.attrs['beta'] != 1.0:
                return None
        if 'tranB' in node.attrs:
            if node.attrs['transB'] != 1:
                return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        weight_buffer = memory_manager.get_buffer(graph, node.inputs[1])
        bias_buffer = None
        if len(node.inputs) > 2:
            bias_buffer = memory_manager.get_buffer(graph, node.inputs[2])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_size = reduce_mult(input_buffer.shape)
        output_size = reduce_mult(output_buffer.shape)

        operation = cls(node, graph)

        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['input_size'] = input_size
        operation.attributes['weight_buffer'] = weight_buffer
        operation.attributes['bias_buffer'] = bias_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['output_size'] = output_size

        return operation


OperationRegistry.register(FullyConnected)


class MaxPool2D(BaseLayer):
    """
    2-dimensional max-pooling operation.
    Kernel size defines the amount of entries from which the maximum will be chosen.
    """
    name = "PicoCNNMaxPool2D"
    operator = "MaxPool"
    template_file = "pico_cnn_max_pool2d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        attrs = node.attrs

        # assert tuple(attrs["pads"]) == (0, 0)  # TODO Check if we need this assertion

        kernel_shape = attrs["kernel_shape"]
        if not (len(kernel_shape) == 2):
            print("{} is not a 2DMaxPool".format(node.name))
            return None

        input_id = node.inputs[0]
        input_shape = graph.get_shape(input_id)

        # input_buffer = "buffer" + input_id
        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])

        num_input_channels = input_shape[1]

        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        kernel_size = attrs["kernel_shape"][0]
        kernel_stride = attrs["strides"][0]

        padding = attrs["pads"]
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        operation = cls(node, graph)

        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['input_height'] = input_shape[2]
        operation.attributes['input_width'] = input_shape[3]
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['kernel_size'] = kernel_size
        operation.attributes['kernel_stride'] = kernel_stride
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding

        return operation


OperationRegistry.register(MaxPool2D)


class MaxPool1D(BaseLayer):
    name = "PicoCNNMaxPool1D"
    operator = "MaxPool"
    template_file = "pico_cnn_max_pool1d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        # assert tuple(attrs["pads"]) == (0, 0)
        kernel_shape = attrs["kernel_shape"]

        if not (len(kernel_shape) == 1 or (len(kernel_shape) == 2 and kernel_shape[1] == 1)):
            print("{} is not a 1DMaxPool".format(node.name))
            return None

        input_id = node.inputs[0]
        input_shape = graph.get_shape(input_id)
        # input_buffer = "buffer" + input_id
        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        num_input_channels = input_shape[1]

        # output_buffer = "buffer" + node.outputs[0]
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        if graph.is_output(node.outputs[0]):
            output_buffer = "output" + node.outputs[0]

        # output_width = graph.get_shape(node.outputs[0], node)[2]

        kernel_size = attrs["kernel_shape"][0]
        kernel_stride = attrs["strides"][0]

        padding = attrs["pads"]
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        input_buffer_size = reduce_mult(input_shape)

        operation = cls(node, graph)

        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['input_width'] = input_shape[2]
        operation.attributes['kernel_size'] = kernel_size
        operation.attributes['kernel_stride'] = kernel_stride
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding

        return operation


OperationRegistry.register(MaxPool1D)


class Relu(BaseLayer):
    """
    Rectified linear unit activation function.
    """
    name = "PicoCNNRelu"
    operator = "Relu"
    template_file = "pico_cnn_relu.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        print("generating relu layer")

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shape = input_buffer.shape

        if len(input_shape) == 2:
            num_input_channels = 1
            input_height = input_shape[0]
            input_width = input_shape[1]
        elif len(input_shape) == 3:
            num_input_channels = input_shape[1]
            input_height = 1
            input_width = input_shape[2]
        elif len(input_shape) == 4:
            num_input_channels = input_shape[1]
            input_height = input_shape[2]
            input_width = input_shape[3]
        else:
            print("ERROR: Unsupported input shape for relu layer: {}".format(input_shape))
            return None

        operation = cls(node, graph)

        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['input_height'] = input_height
        operation.attributes['input_width'] = input_width
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(Relu)


class BatchNorm(BaseLayer):
    name = "PicoCNNBatchNorm"
    operator = "BatchNormalization"
    template_file = "pico_cnn_batchnorm.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        gamma_buffer = memory_manager.get_buffer(graph, node.inputs[1])
        bias_buffer = memory_manager.get_buffer(graph, node.inputs[2])
        mean_buffer = memory_manager.get_buffer(graph, node.inputs[3])
        variance_buffer = memory_manager.get_buffer(graph, node.inputs[4])

        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shape = input_buffer.shape
        num_input_channels = input_shape[1]
        input_height = input_shape[2]
        input_width = 1
        if len(input_shape) >= 4:
            input_width = input_shape[3]

        operation = cls(node, graph)
        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['input_height'] = input_height
        operation.attributes['input_width'] = input_width
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['gamma_buffer'] = gamma_buffer
        operation.attributes['bias_buffer'] = bias_buffer
        operation.attributes['mean_buffer'] = mean_buffer
        operation.attributes['variance_buffer'] = variance_buffer
        operation.attributes['eps'] = attrs['epsilon']

        return operation


OperationRegistry.register(BatchNorm)
#
#
# class Clip(BaseLayer):
#     name = "PicoCNNClip"
#     operator = "Clip"
#     template_file = "pico_cnn_clip.c"
#
#     @classmethod
#     def create(cls, node, graph, memory_manager):
#         print("generating clip layer")
#
#         attrs = node.attrs
#         input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
#         output_buffer = memory_manager.get_buffer(graph, node.outputs[0])
#
#         input_shape = input_buffer.shape
#         input_channels = input_shape[1]
#         input_height = input_shape[2]
#         input_width = 1
#         if len(input_shape) >= 4:
#             input_width = input_shape[3]
#
#         operation = cls(node, graph)
#
#         operation.attributes['input_channels'] = input_channels
#         operation.attributes['input_buffer'] = input_buffer
#         operation.attributes['input_height'] = input_height
#         operation.attributes['input_width'] = input_width
#         operation.attributes['output_buffer'] = output_buffer
#         operation.attributes['max'] = attrs['max']
#         operation.attributes['min'] = attrs['min']
#
#         return operation
#
#
# OperationRegistry.register(Clip)
#
#
# class MatMul(BaseLayer):
#     name = "PicoCNNMatMul"
#     operator = "MatMul"
#     template_file = "pico_cnn_matmul.c"
#
#     @classmethod
#     def create(cls, node, graph, memory_manager):
#         attrs = node.attrs
#
#         input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
#         weight_buffer = memory_manager.get_buffer(graph, node.inputs[1])
#         output_buffer = memory_manager.get_buffer(graph, node.outputs[0])
#
#         input_size = reduce_mult(input_buffer.shape)
#         output_size = reduce_mult(output_buffer.shape)
#
#         operation = cls(node, graph)
#
#         operation.attributes['input_buffer'] = input_buffer
#         operation.attributes['input_size'] = input_size
#         operation.attributes['weight_buffer'] = weight_buffer
#         operation.attributes['output_buffer'] = output_buffer
#         operation.attributes['output_size'] = output_size
#
#         return operation
#
#
# OperationRegistry.register(MatMul)


class AveragePool2D(BaseLayer):
    name = "PicoCNNAveragePool"
    operator = "AveragePool"
    template_file = "pico_cnn_average_pool2d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        kernel_shape = attrs['kernel_shape']

        if not len(kernel_shape) == 2:
            print("{} is not a 2DAvgPool".format(node.name))
            return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])
        bias_buffer = None

        input_shape = input_buffer.shape
        kernel_size = attrs["kernel_shape"][0]
        kernel_stride = attrs["strides"][0]

        num_input_channels = input_shape[1]

        padding = attrs["pads"]
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        count_include_pad = attrs.get("count_include_pad", 0)

        operation = cls(node, graph)

        operation.attributes["num_input_channels"] = num_input_channels
        operation.attributes["input_buffer"] = input_buffer
        operation.attributes["input_height"] = input_shape[2]
        operation.attributes["input_width"] = input_shape[3]
        operation.attributes["output_buffer"] = output_buffer
        operation.attributes["kernel_size"] = kernel_size
        operation.attributes["kernel_stride"] = kernel_stride
        operation.attributes["bias_buffer"] = bias_buffer
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding
        operation.attributes['count_include_pad'] = count_include_pad

        return operation


OperationRegistry.register(AveragePool2D)


class AveragePool1D(BaseLayer):
    name = "PicoCNNAveragePool"
    operator = "AveragePool"
    template_file = "pico_cnn_average_pool1d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        kernel_shape = attrs["kernel_shape"]
        if not (len(kernel_shape) == 1 or (len(kernel_shape) == 2 and kernel_shape[1] == 1)):
            print("{} is not a 1DAvgPool".format(node.name))
            return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])
        bias_buffer = None

        input_shape = input_buffer.shape

        kernel_size = kernel_shape[0]
        kernel_stride = attrs["strides"][0]

        num_input_channels = input_shape[1]

        padding = attrs["pads"]
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        count_include_pad = attrs.get("count_include_pad", 0)

        operation = cls(node, graph)

        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['input_width'] = input_buffer.shape[2]
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['output_width'] = output_buffer.shape[2]
        operation.attributes['kernel_size'] = kernel_size
        operation.attributes['kernel_stride'] = kernel_stride
        operation.attributes['bias_buffer'] = bias_buffer
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding
        operation.attributes['count_include_pad'] = count_include_pad

        return operation


OperationRegistry.register(AveragePool1D)


class GlobalAveragePool2D(BaseLayer):
    name = "PicoCNNGlobalAveragePool"
    operator = "GlobalAveragePool"
    template_file = "pico_cnn_average_pool2d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        # if not len(kernel_shape) == 2:
        #     print("{} is not a 2DAvgPool".format(node.name))
        #     return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])
        bias_buffer = None

        input_shape = input_buffer.shape
        kernel_size = input_shape[2]
        kernel_stride = 1

        num_input_channels = input_shape[1]

        padding = None
        padding_needed = False

        # Only needed as long as we use the regular average_pool2d implementation
        # as we want to compute the average over all pixels
        count_include_pad = 1

        operation = cls(node, graph)

        operation.attributes["num_input_channels"] = num_input_channels
        operation.attributes["input_buffer"] = input_buffer
        operation.attributes["input_height"] = input_shape[2]
        operation.attributes["input_width"] = input_shape[3]
        operation.attributes["output_buffer"] = output_buffer
        operation.attributes["kernel_size"] = kernel_size
        operation.attributes["kernel_stride"] = kernel_stride
        operation.attributes["bias_buffer"] = bias_buffer
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding
        operation.attributes['count_include_pad'] = count_include_pad

        return operation


OperationRegistry.register(GlobalAveragePool2D)


#
#
# class Transpose(BaseLayer):
#     name = "TransposeGeneric"
#     operator = "Transpose"
#     template_file = "transpose.c"
#
#     @classmethod
#     def create(cls, node, graph, memory_manager):
#         attrs = node.attrs
#         input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
#         output_buffer = memory_manager.get_buffer(graph, node.outputs[0])
#
#         input_def = "float_t (*x)[" + "][".join((str(x) for x in input_buffer.shape[1:])) + "]";
#         output_def = "float_t (*y)[" + "][".join((str(x) for x in output_buffer.shape[1:])) + "]";
#
#         input_cast = "float (*)[" + "][".join((str(x) for x in input_buffer.shape[1:])) + "]"
#         output_cast = "float (*)[" + "][".join((str(x) for x in output_buffer.shape[1:])) + "]"
#
#         permutations = attrs['perm']
#         transpose_code = ""
#         print(input_buffer.shape)
#         for input_dim, output_dim in enumerate(permutations):
#             dim_size = input_buffer.shape[input_dim] if input_dim < len(input_buffer.shape) else 1
#             transpose_code += "  " * (input_dim + 1)
#             transpose_code += "for(int dim{} = 0; dim{} < {}; dim{}++)".format(input_dim, input_dim, dim_size,
#                                                                                input_dim)
#             transpose_code += "\n"
#
#         transpose_code += "  " * len(permutations)
#         transpose_code += "    " + "y[" + "][".join(("dim" + str(x) for x in permutations)) + "] = x[" + "][".join(
#             ("dim" + str(x) for x in range(len(input_buffer.shape)))) + "];"
#
#         operation = cls(node, graph)
#
#         operation.attributes['input_buffer'] = input_buffer
#         operation.attributes['output_buffer'] = output_buffer
#         operation.attributes['input_def'] = input_def
#         operation.attributes['output_def'] = output_def
#         operation.attributes['input_cast'] = input_cast
#         operation.attributes['output_cast'] = output_cast
#         operation.attributes['transpose_code'] = transpose_code
#
#         return operation
#
#
# OperationRegistry.register(Transpose)


class Reshape(BaseLayer):
    """
    Transposes the input and writes it to the output buffer.
    """
    name = "ReshapeGeneric"
    operator = "Reshape"
    template_file = "reshape.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        attrs = node.attrs
        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_id = node.inputs[0]
        input_shape = graph.get_shape(input_id)

        output_id = node.outputs[0]
        output_shape = graph.get_shape(output_id)

        assert(reduce_mult(input_shape) == reduce_mult(output_shape))

        if len(input_shape) == 4:
            num_input_channels = input_shape[1]
            input_height = input_shape[2]
            input_width = input_shape[3]
        elif len(input_shape) == 3:
            num_input_channels = input_shape[1]
            input_height = 1
            input_width = input_shape[2]
        else:
            print("ERROR: Unsupported input shape for reshape layer: {}".format(input_shape))
            return None

        operation = cls(node, graph)
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_height'] = input_height
        operation.attributes['input_width'] = input_width

        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(Reshape)


class Flatten(BaseLayer):
    """
    Transposes the input and writes it to the output buffer.
    """
    name = "FlattenGeneric"
    operator = "Flatten"
    template_file = "flatten.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        attrs = node.attrs
        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_id = node.inputs[0]
        input_shape = graph.get_shape(input_id)

        output_id = node.outputs[0]
        output_shape = graph.get_shape(output_id)

        assert(reduce_mult(input_shape) == reduce_mult(output_shape))

        if len(input_shape) == 4:
            num_input_channels = input_shape[1]
            input_height = input_shape[2]
            input_width = input_shape[3]
            no_change = 0
        elif len(input_shape) == 2:
            num_input_channels = 1
            input_height = 1
            input_width = input_shape[1]
            no_change = 1
        else:
            print("ERROR: Unsupported tensor shape in flatten operation: {}".format(input_shape))
            return 1

        operation = cls(node, graph)
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['num_input_channels'] = num_input_channels
        operation.attributes['input_height'] = input_height
        operation.attributes['input_width'] = input_width

        operation.attributes['output_buffer'] = output_buffer

        operation.attributes['no_change'] = no_change

        return operation


OperationRegistry.register(Flatten)


class Add(BaseLayer):
    name = "AddGeneric"
    operator = "Add"
    template_file = "pico_cnn_add.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs
        input_buffers = [memory_manager.get_buffer(graph, i) for i in node.inputs]
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shapes = [b.shape for b in input_buffers]
        for shape in input_shapes:
            if shape != input_shapes[0]:
                print("Broadcasting is not supported for add operation")
                return None

        num_channels = input_shapes[0][1]
        height = input_shapes[0][2]
        width = input_shapes[0][3]

        operation = cls(node, graph)
        operation.attributes['input_buffers'] = input_buffers
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['num_channels'] = num_channels
        operation.attributes['height'] = height
        operation.attributes['width'] = width

        return operation


OperationRegistry.register(Add)

#
# class Sum(Add):
#     name = "SumGeneric"
#     operator = "Sum"
#     template_file = "pico_cnn_add.c"
#
#
# OperationRegistry.register(Sum)


class Softmax(BaseLayer):
    """
    Softmax function.
    """
    name = "SoftmaxGeneric"
    operator = "Softmax"
    template_file = "pico_cnn_softmax.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        attrs = node.attrs
        input_buffers = [memory_manager.get_buffer(graph, i) for i in node.inputs]
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shape = input_buffers[0].shape
        input_height = input_shape[0]
        input_width = input_shape[1]

        operation = cls(node, graph)
        operation.attributes["input_buffer"] = input_buffers[0]
        operation.attributes["output_buffer"] = output_buffer
        operation.attributes["input_height"] = input_height
        operation.attributes["input_width"] = input_width

        return operation


OperationRegistry.register(Softmax)


class LocalResponseNormalization(BaseLayer):
    """
    Local response normalization function.
    """
    name = "LocalResponseNormalizationGeneric"
    operator = "LRN"
    template_file = "pico_cnn_lrn.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        """
        Derive necessary information from ComputeNode, ComputeGraph and MemoryManager to generate the layer code.
        :param node: ComputeNode object of a CNN layer
        :param graph: ComputeGraph object of the CNN
        :param memory_manager: MemoryManager object containing information about input and output buffers.
        :return:
        """
        attrs = node.attrs
        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shape = input_buffer.shape
        height = input_shape[2]
        width = input_shape[3]

        depth = input_shape[1]

        operation = cls(node, graph)
        operation.attributes["input_buffer"] = input_buffer
        operation.attributes["output_buffer"] = output_buffer
        operation.attributes["height"] = height
        operation.attributes["width"] = width
        operation.attributes["depth"] = depth
        operation.attributes["alpha"] = attrs["alpha"]
        operation.attributes["beta"] = attrs["beta"]
        operation.attributes["size"] = attrs["size"]

        return operation


OperationRegistry.register(LocalResponseNormalization)
