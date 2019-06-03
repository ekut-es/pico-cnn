import onnx
from onnx.tools import net_drawer

import argparse
from typing import Text


def import_model(model_path):  # type: (Text) -> onnx.ModelProto
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def create_plot(graph, output):  # type: (onnx.GraphProto, Text) -> None
    pydot_graph = net_drawer.GetPydotGraph(
        graph,
        name=graph.name,
        rankdir='LR',
        node_producer=net_drawer.GetOpNodeProducer(embed_docstring=True, **net_drawer.OP_STYLE),
    )
    pydot_graph.write_dot(output + ".dot")
    pydot_graph.write_svg(output + ".svg")


def main():
    parser = argparse.ArgumentParser(description="Tool for importing ONNX models")
    parser.add_argument(
        "--input",
        type=Text, required=True,
        help="Path to the model.onnx input file.",
    )
    parser.add_argument(
        "--output",
        type=Text, required=False,
        help="Optional output file name (.dot/.svg)."
    )
    args = parser.parse_args()

    model = import_model(args.input)

    if args.output:
        print("Graph of ONNX model will be saved to:", args.output + ".dot", "and", args.output + ".svg")
        create_plot(model.graph, args.output)

    return 0


if __name__ == '__main__':
    main()
