import argparse
from typing import Text
import os
import struct
import glob

import onnx
from onnx import numpy_helper

__author__ = "Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


def main():

    parser = argparse.ArgumentParser(
        description="Tool to parse the reference input and output files from the onnx model zoo.")
    parser.add_argument(
        "--input",
        type=Text,
        required=True,
        help="Path to reference files."
    )
    args = parser.parse_args()

    reference_data_dir = args.input

    print("Using reference data from:", reference_data_dir)

    # Load inputs
    inputs_num = len(glob.glob(os.path.join(reference_data_dir, 'input_*.pb')))

    if inputs_num == 0:
        print("ERROR: There seems to be no *.pb file in the specified directory.")
        exit(1)

    for i in range(inputs_num):
        input_file = os.path.join(reference_data_dir, 'input_{}.pb'.format(i))
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
            packed_file.append(struct.pack('i', tensor_shape[0]))
            packed_file.append(struct.pack('i', tensor_shape[1]))
            packed_file.append(struct.pack('i', tensor_shape[2]))
            packed_file.append(struct.pack('i', tensor_shape[3]))

            for channel in tensor_array[0]:
                for row in channel:
                    packed_file.append(struct.pack('f'*len(row), *row))

            tupac = bytes("end\n", "ascii")
            packed_file.append(struct.pack('{}s'.format(len(tupac)), tupac))

            with open(os.path.join(args.input, "input_{}.data".format(i)), "wb") as f:
                for packed_struct in packed_file:
                    f.write(packed_struct)

            print("Input data shape: {}".format(tensor_shape))

    # Load reference outputs
    ref_outputs_num = len(glob.glob(os.path.join(reference_data_dir, 'output_*.pb')))

    if ref_outputs_num == 0:
        print("ERROR: There seems to be no *.pb file in the specified directory.")
        exit(1)

    for i in range(ref_outputs_num):
        output_file = os.path.join(reference_data_dir, 'output_{}.pb'.format(i))
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
            packed_file.append(struct.pack('i', tensor_shape[0]))
            packed_file.append(struct.pack('i', tensor_shape[1]))

            packed_file.append(struct.pack('f'*len(tensor_array[0]), *tensor_array[0]))

            tupac = bytes("end\n", "ascii")
            packed_file.append(struct.pack('{}s'.format(len(tupac)), tupac))

            with open(os.path.join(args.input, "output_{}.data".format(i)), "wb") as f:
                for packed_struct in packed_file:
                    f.write(packed_struct)

            print("Output data shape: {}".format(tensor_shape))


if __name__ == '__main__':
    main()
