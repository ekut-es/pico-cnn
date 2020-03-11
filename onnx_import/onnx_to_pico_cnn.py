import argparse

import onnx
from onnx import optimizer, utils

from pico_cnn import *
from compute_graph import *
from backend import Backend

__author__ = "Christoph Gerum, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


def onnx_to_pico_cnn(onnx_model, model_name):

    # print(onnx_model.graph)
    # Set input batch size to 1
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "?"
    # onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
    # inputs = onnx_model.graph.input
    # for input in inputs:
    #     dim1 = input.type.tensor_type.shape.dim[0]
    #     dim1.dim_value = 1

    # Remove dropouts
    for op_id, op in enumerate(onnx_model.graph.node):
        if op.op_type == "Dropout":
            op.attribute[0].f = 0.0

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    print("Running model optimization")
    # TODO: There are more optimizations available
    optimized_model = optimizer.optimize(onnx_model, ["eliminate_nop_dropout"])
    optimized_model = utils.polish_model(optimized_model)

    try:
        os.makedirs("./polished_models")
        print("Created directory for polished models.")
    except FileExistsError:
        pass

    onnx.save(optimized_model, os.path.join("./polished_models", "{}_polished.onnx".format(model_name)))

    backend_model = Backend.prepare(optimized_model, model_name)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Tool for converting ONNX models into pico-cnn networks.")
    parser.add_argument(
        "--input",
        type=Text, required=True,
        help="Path to the model.onnx input file.",
    )
    args = parser.parse_args()

    onnx_model = onnx.load(args.input)

    file_name = args.input.split("/")[-1]
    model_name = file_name.split(".")[0]
    print("Generating Pico-CNN Code for model: {}".format(model_name))

    onnx_to_pico_cnn(onnx_model, model_name)

    return 0


if __name__ == '__main__':
    main()
