import argparse
from typing import Text
import os
import glob

import numpy as np
import onnx
import caffe2.python.onnx.backend as backend
# import caffe2
# from caffe2.python import *
import torch
import torchvision

import warnings
from onnx_tf.backend import prepare

from onnx import numpy_helper
from onnx import helper
from onnx import TensorProto

from PIL import Image


intermediate = []


def hook(module, input, output):
    intermediate.append(output)


def main():

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Tool to compare output of onnx model to sample data.")
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
    # inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    # for i in range(inputs_num):
    #     input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    #     tensor = onnx.TensorProto()
    #     with open(input_file, 'rb') as f:
    #         tensor.ParseFromString(f.read())
    #     inputs.append(numpy_helper.to_array(tensor))
    #
    # # Load reference outputs
    # ref_outputs = []
    # ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    # for i in range(ref_outputs_num):
    #     output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    #     tensor = onnx.TensorProto()
    #     with open(output_file, 'rb') as f:
    #         tensor.ParseFromString(f.read())
    #     ref_outputs.append(numpy_helper.to_array(tensor))

    more_outputs = []
    output_to_check = []

    more_outputs.append(helper.make_tensor_value_info('conv5_2', TensorProto.FLOAT, (1, 256, 12, 12)))
    more_outputs.append(helper.make_tensor_value_info('pool5_1', TensorProto.FLOAT, (1, 256, 6, 6)))
    output_to_check.append('conv5_2')
    output_to_check.append('pool5_1')
    model.graph.output.extend(more_outputs)

    img = Image.open("/home/junga/Programming/pico-cnn/onnx_import/generated_code/alexnet_trained/227x227_scale_ILSVRC2012_val_00000018.JPEG")
    img.load()
    data = np.asarray(img, dtype=float)

    data2 = np.ones((1, 3, 224, 224), dtype=float).astype(np.float32)

    # for i, col in enumerate(data):
    #     for j, row in enumerate(col):
    #         for k, channel in enumerate(row):
    #             data2[0][k][j][i] = data[i][j][k]

    # inputs.append(data.reshape((1,3,227,227)))
    inputs.append(data2)

    tf_rep = prepare(model, device='CPU')
    print(tf_rep.tensor_dict)
    tf_out = tf_rep.run(inputs)

    for op in output_to_check:
        np.savetxt(op.replace("/", "__") + ".tf", tf_out[op].flatten(), delimiter='\t')

    # Run the model on the backend
    outputs = list(backend.run_model(model, inputs, device='CPU'))

    np.savetxt("conv5_2.cf2", outputs[1].flatten(), delimiter='\t')
    np.savetxt("pool5_1.cf2", outputs[2].flatten(), delimiter='\t')

    # Compare the results with reference outputs.
    for ref_o, o in zip(ref_outputs, outputs):
        np.testing.assert_almost_equal(ref_o, o, decimal=4)

    output_tf = list(tf_out['prob_1'])
    for ref_o, o in zip(ref_outputs[0], output_tf):
        np.testing.assert_almost_equal(ref_o, o, decimal=4)

    print("Reference output matches computed output of network.")


if __name__ == '__main__':
    main()
