import argparse
from typing import Text

import onnx

from onnx_importer import import_model
from data_structures import *


def onnx_to_pico_cnn(onnx_model):
    graph = ComputeGraph.from_onnx(onnx_model.graph)
    for node in graph.nodes:
        print(node.name)
        print(node.op_type)
        print(node.attrs)
        print(node.inputs)
        print(node.outputs)
        print()


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
