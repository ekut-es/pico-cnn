from ir import *
#from utils import reduce_mult

from jinja2 import Environment, FileSystemLoader

import numpy as np

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")

template_env = Environment(loader=FileSystemLoader(template_dir))


class BaseLayer(object):
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
    name = "PicoCNNConv2D"
    operator = "Conv"
    template_file = "conv2d.c"

    @classmethod
    def create(cls, node, graph, memory_manager):
        operation = cls(node, graph)

        attrs = node.attrs

        operation.attributes['num_out_channels'] = 20
        operation.attributes['input_buffer'] = "t10k_images"
        operation.attributes['input_height'] = 28
        operation.attributes['input_width'] = 28
        operation.attributes['output_buffer'] = "c1_output"
        operation.attributes['kernel'] = "c1_kernels"
        operation.attributes['kernel_size'] = 5
        operation.attributes['stride'] = 1
        operation.attributes['padding'] = 0
        operation.attributes['bias'] = "c1_bias"

        return operation

OperationRegistry.register(Conv2D)