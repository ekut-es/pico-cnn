from collections import namedtuple
from typing import Any, Text, Iterable, List, Dict, Sequence, Optional, Tuple, Union
from onnx import AttributeProto, numpy_helper
import numpy as np


def _convertAttributeProto(onnx_arg): 
    """
    Convert an ONNX AttributeProto into an appropriate Python object for the type.
    :param onnx_arg: A node of the graph representing the CNN
    :return: Tensor attribute gets returned as numpy array
    """
    if onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        return numpy_helper.to_array(onnx_arg.t)
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


EdgeInfo = namedtuple('EdgeInfo', ['name', 'type', 'shape'])


def _input_from_onnx_input(input) -> EdgeInfo:
    """
    Create EdgeInfo named tupel containing name, type and shape of the input.
    :param input: Input or output of the graph representation of the CNN.
    :return: EdgeInfo tupel
    """
    name = input.name
    type = input.type.tensor_type.elem_type
    shape = tuple([d.dim_value for d in input.type.tensor_type.shape.dim])
    return EdgeInfo(name, type, shape)


class Attributes(Dict[Text, Any]):
    """
    Custom dictionary object containing the parsed information from the protobuf attributes.
    """
    @staticmethod
    def from_onnx(args: AttributeProto) -> Any:
        d = Attributes()
        for arg in args:
            d[arg.name] = _convertAttributeProto(arg)
        return d


class ComputeNode(object):
    """
    Contains all important information of a node in the graph representing the CNN.
    """
    def __init__(self, name: str,
                 op_type: str,
                 attrs: Attributes,
                 inputs: List[str],
                 outputs: List[str]) -> None:
        self.name: str = name
        self.op_type: str = op_type
        self.attrs: Attributes = attrs
        self.inputs: List[str] = inputs
        self.outputs: List[str] = outputs

        self.input_tensors: Dict[str, np.ndarray] = {}
        self.parents: List[ComputeNode] = []
        self.children: List[ComputeNode] = []
        self.metadata: Dict[Any, Any] = {}
    
    @staticmethod
    def from_onnx(node) -> Any:  
        attrs = Attributes.from_onnx(node.attribute)
        name = Text(node.name)
        if len(name) == 0:
            name = node.op_type + "_".join(node.output)
        return ComputeNode(name, node.op_type, attrs, list(node.input), list(node.output))


class ComputeGraph(object):
    """
    Graph representing the CNN.
    """
    def __init__(self, nodes: List[ComputeNode], inputs, outputs, shape_dict):
        self.nodes: List[ComputeNode] = nodes
        self.inputs = inputs
        self.outputs = outputs
        self.shape_dict: Dict[str, np.ndarray] = shape_dict

    @staticmethod
    def from_onnx(graph) -> Any:
        """
        Create the ComputeGraph from the onnx model.
        :param graph: The graph stored in the onnx file.
        :return: ComputeGraph representing the CNN.
        """
        input_tensors = {
            t.name: numpy_helper.to_array(t) for t in graph.initializer
        }
       
        nodes_ = []
        nodes_by_input: Dict[str, List[ComputeNode]] = {}
        nodes_by_output: Dict[str, ComputeNode] = {}
        for node in graph.node:
            node_ = ComputeNode.from_onnx(node)
            for input_ in node_.inputs:
                if input_ in input_tensors:
                    node_.input_tensors[input_] = input_tensors[input_]
                else:
                    if input_ in nodes_by_input:
                        input_nodes = nodes_by_input[input_]
                    else:
                        input_nodes = []
                        nodes_by_input[input_] = input_nodes
                    input_nodes.append(node_)
            for output_ in node_.outputs:
                nodes_by_output[output_] = node_
            nodes_.append(node_)

        inputs = []
        for i in graph.input:
            if i.name not in input_tensors:
                inputs.append(_input_from_onnx_input(i))

        outputs = []
        for o in graph.output:
            outputs.append(_input_from_onnx_input(o))

        for node_ in nodes_:
            for input_ in node_.inputs:
                if input_ in nodes_by_output:
                    node_.parents.append(nodes_by_output[input_])
            for output_ in node_.outputs:
                if output_ in nodes_by_input:
                    node_.children.extend(nodes_by_input[output_])

        # Dictionary to hold the "value_info" field from ONNX graph
        shape_dict: Dict[Text, Any] = {}

        def extract_value_info(shape_dict, 
                               value_info, 
                               ):
           
            t = tuple([int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
            if t:
                shape_dict[value_info.name] = t

        for value_info in graph.value_info:
            extract_value_info(shape_dict, value_info)

        return ComputeGraph(nodes_, inputs, outputs, shape_dict)

    def remove_node(self, node):
        print("Removing node", node.name)
        if not node in self.nodes:
            return
     
        self.nodes.remove(node)
     
        parents = node.parents
     
        for parent in node.parents:
            parent.children.remove(node)
            if not parent.children:
                self.remove_node(parent)
     
        for child in node.children:
            child.parents.remove(node)

    def get_shape(self, name: Text) -> Iterable[int]:
        for input in self.inputs:
            if input.name == name:
                return input.shape
            
        for output in self.outputs:
            if output.name == name:
                return output.shape

        for node in self.nodes:
            if name in node.input_tensors:
                return node.input_tensors[name].shape
        
        if name in self.shape_dict:
            return self.shape_dict[name]
     
        return ()

    def is_input(self, name : Text) -> bool:
        for input in self.inputs:
            if input.name == name:
                return True
     
        return False

    def is_output(self, name : Text) -> bool:
        for output in self.outputs:
            if output.name == name:
                return True
     
        return False

    def is_tensor(self, name : Text) -> bool:
        for node in self.nodes:
            if name in node.input_tensors:
                return True
        return False
