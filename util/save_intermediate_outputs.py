import argparse
from typing import Text
import os
from collections import namedtuple

import numpy as np

import warnings
from onnx_tf.backend import prepare

import onnx
from onnx import helper
from onnx import TensorProto

__author__ = "Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"

IntermediateOutput = namedtuple('IntermediateOutput', ['name', 'shape'])


def intermediate_output(s):
    tmp = s.split(',')
    name = tmp[0]
    shape = tuple(tmp[1:])
    return IntermediateOutput(name, shape)


def main():

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="Debugging-Tool to save output of any layer in a model to text files.")
    parser.add_argument(
        "--input",
        type=Text,
        required=True,
        help="Path to onnx model."
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs='+',
        required=True,
        help="Required input shape. Format \"--shape 1 3 224 224\" for (1, 3, 224, 224) with "
             "(batch_size, num_input_ch, height, width)."
    )
    parser.add_argument(
        "--out_to_check",
        type=intermediate_output,
        nargs='+',
        help="Name and shape of the intermediate outputs that will be saved to files, \
             separated by spaces. E.g. pool5_1,1,256,6,6 fc8_1,1,1000"
    )
    args = parser.parse_args()

    input_path = args.input
    print("Using model:", input_path)

    input_file_name = os.path.basename(input_path)

    model = onnx.load(input_path)

    inputs = []
    input_shape = tuple(args.input_shape)
    data = np.ones(input_shape, dtype=float).astype(np.float32)
    inputs.append(data)

    more_outputs = []
    output_to_check = []

    intermediate_outputs = args.out_to_check

    for out in intermediate_outputs:
        more_outputs.append(helper.make_tensor_value_info(out.name, TensorProto.FLOAT, out.shape))
        output_to_check.append(out.name)

    model.graph.output.extend(more_outputs)

    tf_rep = prepare(model, device='CPU')
    print(tf_rep.tensor_dict)
    tf_out = tf_rep.run(inputs)

    for op in output_to_check:
        np.savetxt("{}_{}".format(input_file_name.split('.')[0],
                                  op.replace("/", "_").replace(".", "_").replace(":", "_") + ".tf"),
                   tf_out[op].flatten(), delimiter='\t')


if __name__ == '__main__':
    main()
