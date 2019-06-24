import argparse
from typing import Text

import onnx
from onnx import optimizer, utils

from onnx_importer import import_model

from pico_cnn import *
from ir import *
from compute_graph import *
from constprop import constant_propagation
from backend import Backend, BackendRep


def onnx_to_pico_cnn(onnx_model):

    # Set input batch size to 1
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

    # Remove dropouts
    for op_id, op in enumerate(onnx_model.graph.node):
        if op.op_type == "Dropout":
            op.attribute[0].f = 0.0

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    print("Running model optimization")
    #TODO: There are more optimizations available
    optimized_model = optimizer.optimize(onnx_model, ["eliminate_nop_dropout"])
    optimized_model = utils.polish_model(optimized_model)

    onnx.save(optimized_model, "polished_model.onnx")

    backend_model = Backend.prepare(optimized_model)

    #inputs = []

    #backend_model.run(inputs)

    return 0

    # graph = ComputeGraph.from_onnx(optimized_model.graph)
    # for node in graph.nodes:
    #
    #     if node.op_type == "Conv":
    #         for op in OperationRegistry.get_ops(node.op_type):
    #             impl = op.create(node, graph, None)
    #
    #         implementation_code = impl.generate_code()
    #         implementation_code += "\n"
    #
    #         print(implementation_code)


def main():
    parser = argparse.ArgumentParser(description="Tool for converting ONNX models into pico-cnn networks.")
    parser.add_argument(
        "--input",
        type=Text, required=True,
        help="Path to the model.onnx input file.",
    )
    args = parser.parse_args()

    onnx_model = import_model(args.input)

    onnx_to_pico_cnn(onnx_model)

    return 0


if __name__ == '__main__':
    main()
