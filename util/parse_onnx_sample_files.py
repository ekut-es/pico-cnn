import argparse
from typing import Text
import os
import struct
import glob

import onnx
from onnx import numpy_helper


def main():

    parser = argparse.ArgumentParser(
        description="Tool to parse the sample input and output files from the onnx model zoo.")
    parser.add_argument(
        "--input",
        type=Text,
        required=True,
        help="Path to sample files."
    )
    args = parser.parse_args()

    sample_data_dir = args.input

    print("Using sample data from:", sample_data_dir)

    # Load inputs
    inputs_num = len(glob.glob(os.path.join(sample_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(sample_data_dir, 'input_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            tensor_array = numpy_helper.to_array(tensor)

        if len(tensor_array.shape) != 4 or tensor_array.shape[0] != 1:
            print("ERROR: Unsupported input shape: {}".format(tensor_array.shape))
            return
        else:
            tensor_shape = tensor_array.shape

            packed_file = list(bytes())
            magic_number = bytes("FCI\n", "ascii")
            packed_file.append(struct.pack('{}s'.format(len(magic_number)), magic_number))
            # TODO: Maybe insert file name to identify the model for which the data is intended
            packed_file.append(struct.pack('i', tensor_shape[1]))
            packed_file.append(struct.pack('i', tensor_shape[2]))
            packed_file.append(struct.pack('i', tensor_shape[3]))

            for channel in tensor_array[0]:
                for row in channel:
                    packed_file.append(struct.pack('f'*len(row), *row))

            with open(os.path.join(args.input, "input_{}.data".format(i)), "wb") as f:
                for packed_struct in packed_file:
                    f.write(packed_struct)

            print("Input data shape: {}".format(tensor_shape))

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(sample_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(sample_data_dir, 'output_{}.pb'.format(i))
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            tensor_array = numpy_helper.to_array(tensor)

        if len(tensor_array.shape) != 2 or tensor_array.shape[0] != 1:
            print("ERROR: Unsupported output shape: {}".format(tensor_array.shape))
            return
        else:
            tensor_shape = tensor_array.shape

            packed_file = list(bytes())
            magic_number = bytes("FCO\n", "ascii")
            packed_file.append(struct.pack('{}s'.format(len(magic_number)), magic_number))
            # TODO: Maybe insert file name to identify the model for which the data is intended
            packed_file.append(struct.pack('i', tensor_shape[1]))

            packed_file.append(struct.pack('f'*len(tensor_array[0]), *tensor_array[0]))

            with open(os.path.join(args.input, "output_{}.data".format(i)), "wb") as f:
                for packed_struct in packed_file:
                    f.write(packed_struct)

            print("Output data shape: {}".format(tensor_shape))


if __name__ == '__main__':
    main()
