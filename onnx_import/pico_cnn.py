""" All operator related code will be generated from the corresponding operator classes. """
from ir import *
from utils import reduce_mult

from jinja2 import Environment, FileSystemLoader

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "code_templates")

template_env = Environment(loader=FileSystemLoader(template_dir))

__author__ = "Christoph Gerum, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


class BaseLayer(object):
    """
    Base layer class to inherit from. Operator implementations see below.
    """
    def __init__(self, node, graph):
        print("Generating layer", node.name)
        self.node = node
        self.graph = graph
        self.attributes = {}

    def generate_declaration(self):
        template = template_env.get_template(self.template_file_declaration)
        return template.render(**self.attributes)

    def generate_allocation(self):
        template = template_env.get_template(self.template_file_allocation)
        return template.render(**self.attributes)

    def generate_execution(self):
        template = template_env.get_template(self.template_file_execution)
        return template.render(**self.attributes)

    def generate_deletion(self):
        template = template_env.get_template(self.template_file_deletion)
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
    template_file_declaration = "conv/pico_cnn_conv2d_decl.cpp"
    template_file_allocation = "conv/pico_cnn_conv2d_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

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

        if not (len(input_shape) == 4):
            print("{} is not a 2DConvolution".format(node.name))
            return None

        if input_shape[2] != input_shape[3]:
            print("WARNING: Not a squared input image ({}x{})!!!".format(input_shape[2], input_shape[3]))

        output_shape = output_buffers[0].shape
        output_size = output_shape[2]
        num_output_channels = output_shape[1]

        kernel_shape = attrs["kernel_shape"]
        stride = attrs["strides"]
        # dilation = attrs["dilations"][0]

        num_groups = attrs.get("group", 1)

        # TODO: handle auto padding
        if "auto_pad" in attrs:
            print("{} auto padding is currently not supported".format(node.name))
            exit(1)

        padding = attrs.get("pads", [0, 0, 0, 0])

        if (kernel_shape[0] % 2) == 0 or (kernel_shape[1] % 2) == 0:
            print("PicoCNN only supports odd kernel sizes in 2D convolution")
            exit(1)

        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffers[0]
        operation.attributes['kernel'] = input_buffers[1]
        operation.attributes['stride'] = stride
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding
        operation.attributes['output_buffer'] = output_buffers[0]
        operation.attributes['num_groups'] = num_groups

        if len(input_buffers) > 2:
            operation.attributes['bias_buffer'] = input_buffers[2]

        return operation


OperationRegistry.register(Conv2D)


class Conv1D(BaseLayer):
    name = "PicoCNNConv1D"
    operator = "Conv"
    template_file_declaration = "conv/pico_cnn_conv2d_decl.cpp"
    template_file_allocation = "conv/pico_cnn_conv1d_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        operation = cls(node, graph)

        attrs = node.attrs
        input_buffers = [memory_manager.get_buffer(graph, id) for id in node.inputs]
        output_buffers = [memory_manager.get_buffer(graph, id) for id in node.outputs]

        input_shape = input_buffers[0].shape

        if not (len(input_shape) == 3):
            print("{} is not a 1DConvolution".format(node.name))
            return None

        kernel_shape = attrs["kernel_shape"]
        stride = attrs["strides"]
        # dilation = attrs["dilations"][0]

        num_groups = attrs.get("group", 1)

        # TODO: handle auto padding
        if "auto_pad" in attrs:
            print("{} auto padding is currently not supported".format(node.name))
            exit(1)

        padding = attrs.get("pads", [0, 0])

        if (kernel_shape[0] % 2) == 0:
            print("PicoCNN only supports odd kernel sizes in 2D convolution")
            exit(1)

        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffers[0]
        operation.attributes['kernel'] = input_buffers[1]
        operation.attributes['stride'] = stride
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding
        operation.attributes['output_buffer'] = output_buffers[0]
        operation.attributes['num_groups'] = num_groups

        if len(input_buffers) > 2:
            operation.attributes['bias_buffer'] = input_buffers[2]

        return operation


OperationRegistry.register(Conv1D)


class FullyConnected(BaseLayer):
    """
    Fully-connected layer. The corresponding operator in an onnx model is "Gemm".
    Basically this is a simple matrix x matrix multiplication.
    """
    name = "PicoCNNFullyConnected"
    operator = "Gemm"
    template_file_declaration = "fc/pico_cnn_fc_decl.cpp"
    template_file_allocation = "fc/pico_cnn_fc_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

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

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['weight_buffer'] = weight_buffer
        operation.attributes['bias_buffer'] = bias_buffer
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(FullyConnected)


class MaxPool2D(BaseLayer):
    """
    2-dimensional max-pooling operation.
    Kernel size defines the amount of entries from which the maximum will be chosen.
    """
    name = "PicoCNNMaxPool2D"
    operator = "MaxPool"
    template_file_declaration = "pool/pico_cnn_max_pool2d_decl.cpp"
    template_file_allocation = "pool/pico_cnn_max_pool2d_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

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

        kernel_shape = attrs["kernel_shape"]
        stride = attrs["strides"]

        padding = attrs.get("pads", (0, 0, 0, 0))
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['kernel_shape'] = kernel_shape
        operation.attributes['stride'] = stride
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding

        return operation


OperationRegistry.register(MaxPool2D)


class MaxPool1D(BaseLayer):
    name = "PicoCNNMaxPool1D"
    operator = "MaxPool"
    template_file_declaration = "pool/pico_cnn_max_pool2d_decl.cpp"
    template_file_allocation = "pool/pico_cnn_max_pool1d_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        kernel_shape = attrs["kernel_shape"]
        if not (len(kernel_shape) == 1):
            print("{} is not a 1DMaxPool".format(node.name))
            return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        stride = attrs["strides"]

        padding = attrs.get("pads", (0, 0))
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['kernel_shape'] = kernel_shape
        operation.attributes['stride'] = stride
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
    template_file_declaration = "activation/pico_cnn_relu_decl.cpp"
    template_file_allocation = "activation/pico_cnn_relu_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

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

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(Relu)


class BatchNorm(BaseLayer):
    name = "PicoCNNBatchNorm"
    operator = "BatchNormalization"
    template_file_declaration = "batch_normalization/pico_cnn_batchnorm_decl.cpp"
    template_file_allocation = "batch_normalization/pico_cnn_batchnorm_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        gamma_buffer = memory_manager.get_buffer(graph, node.inputs[1])
        bias_buffer = memory_manager.get_buffer(graph, node.inputs[2])
        mean_buffer = memory_manager.get_buffer(graph, node.inputs[3])
        variance_buffer = memory_manager.get_buffer(graph, node.inputs[4])

        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier

        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['gamma_buffer'] = gamma_buffer
        operation.attributes['bias_buffer'] = bias_buffer
        operation.attributes['mean_buffer'] = mean_buffer
        operation.attributes['variance_buffer'] = variance_buffer
        operation.attributes['eps'] = attrs['epsilon']

        return operation


OperationRegistry.register(BatchNorm)


class Clip(BaseLayer):
    name = "PicoCNNClip"
    operator = "Clip"
    template_file_declaration = "activation/pico_cnn_clip_decl.cpp"
    template_file_allocation = "activation/pico_cnn_clip_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):

        attrs = node.attrs
        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['min'] = attrs['min']
        operation.attributes['max'] = attrs['max']

        return operation


OperationRegistry.register(Clip)


class MatMul(BaseLayer):
    name = "PicoCNNMatMul"
    operator = "MatMul"
    template_file_declaration = "fc/pico_cnn_matmul_decl.cpp"
    template_file_allocation = "fc/pico_cnn_matmul_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        weight_buffer = memory_manager.get_buffer(graph, node.inputs[1])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_size = reduce_mult(input_buffer.shape)
        output_size = reduce_mult(output_buffer.shape)

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['weight_buffer'] = weight_buffer
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(MatMul)


class Mul(BaseLayer):
    name = "PicoCNNMul"
    operator = "Mul"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "tensor_operations/pico_cnn_mul.cpp"
    template_file_deletion = "empty.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        factor = node.input_tensors[node.inputs[1]]

        if factor.size != 1:
            print("ERROR: Mul operation currently does not support Tensor-styled factors.")
            exit(1)

        input_shape = input_buffer.shape
        output_shape = output_buffer.shape

        assert(input_shape == output_shape)

        operation = cls(node, graph)

        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['factor'] = factor

        return operation


OperationRegistry.register(Mul)


class AveragePool2D(BaseLayer):
    name = "PicoCNNAveragePool"
    operator = "AveragePool"
    template_file_declaration = "pool/pico_cnn_avg_pool2d_decl.cpp"
    template_file_allocation = "pool/pico_cnn_avg_pool2d_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        kernel_shape = attrs['kernel_shape']

        if not len(kernel_shape) == 2:
            print("{} is not a 2DAvgPool".format(node.name))
            return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        kernel_stride = attrs["strides"]

        padding = attrs.get("pads", (0, 0, 0, 0))
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        count_include_pad = attrs.get("count_include_pad", 0)

        # As we have to pass the count_include_pad attribute even if we don't have padding we need to
        # set it manually so that the correct amount of pixels is used for the average computation
        if not padding_needed:
            count_include_pad = 1

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes["input_buffer"] = input_buffer
        operation.attributes["output_buffer"] = output_buffer
        operation.attributes["kernel_shape"] = kernel_shape
        operation.attributes["stride"] = kernel_stride
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding
        operation.attributes['count_include_pad'] = count_include_pad

        return operation


OperationRegistry.register(AveragePool2D)


class AveragePool1D(BaseLayer):
    name = "PicoCNNAveragePool"
    operator = "AveragePool"
    template_file_declaration = "pool/pico_cnn_avg_pool2d_decl.cpp"
    template_file_allocation = "pool/pico_cnn_avg_pool1d_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        kernel_shape = attrs["kernel_shape"]

        if not (len(kernel_shape) == 1 or (len(kernel_shape) == 2 and kernel_shape[1] == 1)):
            print("{} is not a 1DAvgPool".format(node.name))
            return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        kernel_stride = attrs["strides"]

        padding = attrs["pads"]
        padding_needed = False
        for num in padding:
            if num != 0:
                padding_needed = True

        count_include_pad = attrs.get("count_include_pad", 0)

        # As we have to pass the count_include_pad attribute even if we don't have padding we need to
        # set it manually so that the correct amount of pixels is used for the average computation
        if not padding_needed:
            count_include_pad = 1

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes["input_buffer"] = input_buffer
        operation.attributes["output_buffer"] = output_buffer
        operation.attributes["kernel_shape"] = kernel_shape
        operation.attributes["stride"] = kernel_stride
        operation.attributes['padding_needed'] = padding_needed
        operation.attributes['padding'] = padding
        operation.attributes['count_include_pad'] = count_include_pad
        return operation


OperationRegistry.register(AveragePool1D)


# class GlobalMaxPool2D(BaseLayer):
#     name = "PicoCNNGlobalMaxPool2D"
#     operator = "GlobalMaxPool"
#     template_file = "pico_cnn_global_max_pool2d.c"
#
#     @classmethod
#     def create(cls, node, graph, memory_manager):
#         attrs = node.attrs
#
#         # if not len(kernel_shape) == 2:
#         #     print("{} is not a 2DAvgPool".format(node.name))
#         #     return None
#
#         input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
#         output_buffer = memory_manager.get_buffer(graph, node.outputs[0])
#
#         input_shape = input_buffer.shape
#         num_input_channels = input_shape[1]
#         input_height = input_shape[2]
#         input_width = input_shape[3]
#
#         operation = cls(node, graph)
#
#         operation.attributes["num_input_channels"] = num_input_channels
#         operation.attributes["input_buffer"] = input_buffer
#         operation.attributes["input_height"] = input_height
#         operation.attributes["input_width"] = input_width
#         operation.attributes["output_buffer"] = output_buffer
#
#         return operation
#
#
# OperationRegistry.register(GlobalMaxPool2D)


class GlobalAveragePool2D(BaseLayer):
    name = "PicoCNNGlobalAveragePool"
    operator = "GlobalAveragePool"
    template_file_declaration = "pool/pico_cnn_global_avg_pool2d_decl.cpp"
    template_file_allocation = "pool/pico_cnn_global_avg_pool2d_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        # if not len(kernel_shape) == 2:
        #     print("{} is not a 2DAvgPool".format(node.name))
        #     return None

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes["input_buffer"] = input_buffer
        operation.attributes["output_buffer"] = output_buffer

        return operation


OperationRegistry.register(GlobalAveragePool2D)


class Transpose(BaseLayer):
    name = "TransposeGeneric"
    operator = "Transpose"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "tensor_operations/transpose.cpp"
    template_file_deletion = "empty.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs
        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        permutations = attrs['perm']
        orig_permutation = range(len(input_buffer.shape))
        transpose_code = ""

        input_shape = input_buffer.shape
        output_shape = output_buffer.shape

        for input_dim, output_dim in enumerate(permutations):
            dim_size = input_buffer.shape[input_dim] if input_dim < len(input_buffer.shape) else 1
            transpose_code += "    " * (input_dim + 1)
            transpose_code += "for(uint32_t dim{} = 0; dim{} < {}; dim{}++)".format(input_dim, input_dim, dim_size,
                                                                                    input_dim)
            transpose_code += "\n"

        transpose_code += "    " * len(permutations)

        if len(permutations) == 4:
            num_input_channels = input_shape[1]
            input_height = input_shape[2]
            input_width = input_shape[3]

            num_output_channels = output_shape[1]
            output_height = output_shape[2]
            output_width = output_shape[3]

            test_code = "{}->access({}, {}, {}, {}, {}, {}, {})".format(output_buffer.name,
                                                                        "dim" + str(permutations[0]),
                                                                        "dim" + str(permutations[1]),
                                                                        "dim" + str(permutations[2]),
                                                                        "dim" + str(permutations[3]),
                                                                        num_output_channels,
                                                                        output_height,
                                                                        output_width)\
                        + " = " \
                        + "{}->access({}, {}, {}, {}, {}, {}, {});".format(input_buffer.name,
                                                                           "dim" + str(orig_permutation[0]),
                                                                           "dim" + str(orig_permutation[1]),
                                                                           "dim" + str(orig_permutation[2]),
                                                                           "dim" + str(orig_permutation[3]),
                                                                           num_input_channels,
                                                                           input_height,
                                                                           input_width)
        elif len(permutations) == 2:
            input_width = input_shape[1]
            output_width = output_shape[1]

            test_code = "{}->access({}, {}, {})".format(output_buffer.name,
                                                        "dim"+str(permutations[0]),
                                                        "dim"+str(permutations[1]),
                                                        output_width) \
                        + " = " \
                        + "{}->access({}, {}, {});".format(input_buffer.name,
                                                           "dim"+str(orig_permutation[0]),
                                                           "dim"+str(orig_permutation[1]),
                                                           input_width)
        else:
            print("ERROR: Unsupported permutation in Transpose operation.")
            exit(1)

        transpose_code += "    " + test_code

        operation = cls(node, graph)
        operation.attributes['transpose_code'] = transpose_code

        return operation


OperationRegistry.register(Transpose)


class Concat(BaseLayer):
    """
    Transposes the input and writes it to the output buffer.
    """
    name = "ConcatGeneric"
    operator = "Concat"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "tensor_operations/pico_cnn_concat.cpp"
    template_file_deletion = "empty.cpp"

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

        if attrs['axis'] != 1:
            print("ERROR: Currently only concatenation along channels is supported!")
            exit(1)

        input_buffers = []
        for input in node.inputs:
            input_buffers.append(memory_manager.get_buffer(graph, input))

        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shapes = [graph.get_shape(input_id) for input_id in node.inputs]
        num_input_shapes = len(input_shapes)
        output_shape = graph.get_shape(node.outputs[0])

        for input_shape in input_shapes:
            assert(len(input_shape) == len(output_shape))
            for i, s in enumerate(input_shape):
                if i != attrs['axis']:
                    assert (s == output_shape[i])

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        input_declaration = "pico_cnn::naive::Tensor* inputs_{}[{}] = {{".format(identifier,
                                                                                 str(num_input_shapes))
        for idx in range(num_input_shapes):
            input_declaration += "{}".format(input_buffers[idx].name)
            if idx < num_input_shapes-1:
                input_declaration += ", "
            else:
                input_declaration += "};"

        dimension = attrs['axis']

        operation = cls(node, graph)

        operation.attributes['input_declaration'] = input_declaration

        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['num_inputs'] = len(input_buffers)
        operation.attributes['inputs'] = "inputs_{}".format(identifier)
        operation.attributes['dimension'] = dimension

        return operation


OperationRegistry.register(Concat)


class Reshape(BaseLayer):
    """
    Transposes the input and writes it to the output buffer.
    """
    name = "ReshapeGeneric"
    operator = "Reshape"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "tensor_operations/pico_cnn_reshape.cpp"
    template_file_deletion = "empty.cpp"

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

        operation = cls(node, graph)
        operation.attributes['name'] = node.name
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(Reshape)


class Flatten(BaseLayer):
    """
    Transposes the input and writes it to the output buffer.
    """
    name = "FlattenGeneric"
    operator = "Flatten"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "tensor_operations/pico_cnn_flatten.cpp"
    template_file_deletion = "empty.cpp"

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

        operation = cls(node, graph)
        operation.attributes['name'] = node.name
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(Flatten)


class Add(BaseLayer):
    name = "AddGeneric"
    operator = "Add"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "pico_cnn_add.cpp"
    template_file_deletion = "empty.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs
        input_buffers = [memory_manager.get_buffer(graph, i) for i in node.inputs]
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shapes = [b.shape for b in input_buffers]
        for shape in input_shapes:
            if shape != input_shapes[0]:
                if len(input_shapes[0]) <= 2 and len(shape) <= 2:
                    print("Broadcasting, but it still works without doing anything...")
                else:
                    # TODO: This needs a better solution!!!
                    print("Broadcasting is not supported for add operation")
                    exit(1)

        operation = cls(node, graph)
        operation.attributes['name'] = node.name
        operation.attributes['input_buffers'] = input_buffers
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(Add)


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
    template_file_declaration = "activation/pico_cnn_softmax_decl.cpp"
    template_file_allocation = "activation/pico_cnn_softmax_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

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

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes["input_buffer"] = input_buffers[0]
        operation.attributes["output_buffer"] = output_buffer

        return operation


OperationRegistry.register(Softmax)


class LocalResponseNormalization(BaseLayer):
    """
    Local response normalization function.
    """
    name = "LocalResponseNormalizationGeneric"
    operator = "LRN"
    template_file_declaration = "activation/pico_cnn_lrn_decl.cpp"
    template_file_allocation = "activation/pico_cnn_lrn_alloc.cpp"
    template_file_execution = "layer_exec.cpp"
    template_file_deletion = "layer_delete.cpp"

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

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['name'] = node.name
        operation.attributes['identifier'] = identifier
        operation.attributes["input_buffer"] = input_buffer
        operation.attributes["output_buffer"] = output_buffer
        operation.attributes["alpha"] = attrs["alpha"]
        operation.attributes["beta"] = attrs["beta"]
        operation.attributes["n"] = attrs["size"]

        return operation


OperationRegistry.register(LocalResponseNormalization)


class Squeeze(BaseLayer):
    """
    Remove single-dimensional entries from the shape of a tensor.
    """
    name = "SqueezeGeneric"
    operator = "Squeeze"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "tensor_operations/pico_cnn_squeeze.cpp"
    template_file_deletion = "empty.cpp"

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

        operation = cls(node, graph)
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer

        return operation


OperationRegistry.register(Squeeze)


class Pad(BaseLayer):
    name = "PicoCNNPad"
    operator = "Pad"
    template_file_declaration = "empty.cpp"
    template_file_allocation = "empty.cpp"
    template_file_execution = "tensor_operations/pico_cnn_pad.cpp"
    template_file_deletion = "empty.cpp"

    @classmethod
    def create(cls, node, graph, memory_manager):
        attrs = node.attrs

        input_buffer = memory_manager.get_buffer(graph, node.inputs[0])
        output_buffer = memory_manager.get_buffer(graph, node.outputs[0])

        input_shape = input_buffer.shape

        pads = attrs['pads']
        mode = attrs['mode'].decode("utf-8")
        value = attrs['value']

        if mode != 'constant':
            print("ERROR: Pad layer only supported in mode constant (not: {}).").format(mode)
            exit(1)

        if len(input_shape) == 4:
            if pads[0] != 0 or pads[1] != 0 or pads[4] != 0 or pads[5] != 0:
                print("ERROR: Padding only supported for height and width dimension. pads = {}").format(pads)
                exit(1)
        elif len(input_shape) == 3:
            if pads[0] != 0 or pads[1] != 0 or pads[3] != 0 or pads[4] != 0:
                print("ERROR: Padding only supported for height and width dimension. pads = {}").format(pads)
                exit(1)
        elif len(input_shape) == 2:
            pass
        else:
            print("ERROR: Unsupported shape for pad operation: {}".format(input_shape))
            exit(1)

        operation = cls(node, graph)

        identifier = node.name.replace('.', '_').replace(':', '_').replace('/', '_')

        operation.attributes['identifier'] = identifier

        operation.attributes['num_dims'] = len(input_shape)
        operation.attributes['input_buffer'] = input_buffer
        operation.attributes['output_buffer'] = output_buffer
        operation.attributes['padding'] = pads
        operation.attributes['initializer'] = value

        return operation


OperationRegistry.register(Pad)
