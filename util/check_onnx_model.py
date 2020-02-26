import argparse
from typing import Text
import os
import glob

import numpy as np

import onnx
from onnx import numpy_helper

import caffe2.python.onnx.backend as backend

import warnings


def main():

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Tool to compare output of onnx model to \
                                                 sample data obtained from the onnx model-zoo.")
    parser.add_argument(
        "--input",
        type=Text,
        required=True,
        help="Path to onnx model."
    )
    args = parser.parse_args()

    input_path = args.input
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), "test_data_set_0")

    print(input_path)
    print(test_data_dir)

    model = onnx.load(input_path)

    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(tensor))

    # Run the model on the backend
    outputs = list(backend.run_model(model, inputs, device='CPU'))

    # Compare the results with reference outputs.
    for ref_o, o in zip(ref_outputs, outputs):
        np.testing.assert_almost_equal(ref_o, o, decimal=4)

    print("Reference output matches computed output of network.")


if __name__ == '__main__':
    main()
