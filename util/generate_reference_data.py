import argparse
from typing import Text
import struct
import os

import numpy as np
import onnx
import caffe2.python.onnx.backend as backend

import magic
from scipy.io.wavfile import read
import imageio

__author__ = "Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"

supported_file_types = ['audio/x-wav', 'image/jpeg', 'image/x-portable-greymap']


def main():
    parser = argparse.ArgumentParser(
        description="This tool creates appropriate sample input and output data for a given neural network. "
                    "This is done by running the given network (onnx model) on matching data "
                    "(supported file types: {}). As backend caffe2 will be used.".format(supported_file_types)
    )
    parser.add_argument(
        "--model",
        type=Text,
        required=True,
        help="Path to the onnx model."
    )
    parser.add_argument(
        "--file",
        type=Text,
        required=False,
        help="Path to the data the network should process. If not provided random values will be used."
    )
    parser.add_argument(
        "--range",
        type=int,
        nargs=2,
        required=False,
        default=[0, 1],
        help="Range for random input values. Default is [0,1)"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs='+',
        required=True,
        help="Required input shape. Format \"--shape 1 1 1 16000\" for (1, 1, 1, 16000) with "
             "(batch_size, num_input_ch, height, width)."
    )
    args = parser.parse_args()

    if args.file:
        input_file = args.file
        input_file_path = os.path.dirname(input_file)
        input_file_name = os.path.basename(input_file)

        # Check file type
        file_type = magic.from_file(input_file, mime=True)
        if file_type not in supported_file_types:
            print("Unsupported file type: {}. At the moment this tool only supports: {}".format(file_type,
                                                                                                supported_file_types))
            exit(1)
    else:
        print("No input file specified. Generating random input.")
        file_type = 'random'
        input_file = "random"
        input_file_name = input_file

    r = args.range

    onnx_model = args.model
    onnx_file_path = os.path.dirname(onnx_model)

    input_shape = tuple(args.shape)

    model = onnx.load(onnx_model)
    onnx.checker.check_model(model)

    packed_input = list(bytes())
    magic_input = bytes("FCI\n", "ascii")

    packed_output = list(bytes())
    magic_output = bytes("FCO\n", "ascii")

    if input_shape[0] != 1:
        print("ERROR: Batch sizes != 1 not supported at the moment.")
        exit(1)
    else:
        print("Input shape from parameters is: {}".format(input_shape))

    # Process input data and write it to our custom binary format.
    if file_type == 'audio/x-wav':
        input_data = read(input_file)
        input_data = np.array(input_data[1], dtype=float)
        input_data = input_data/max(input_data)
        input_data = input_data.astype(np.float32)

        packed_input.append(struct.pack('{}s'.format(len(magic_input)), magic_input))
        packed_input.append(struct.pack('i', input_shape[1]))  # Number of channels
        packed_input.append(struct.pack('i', 1))  # Channel height
        packed_input.append(struct.pack('i', input_shape[2]))  # Channel width
        packed_input.append(struct.pack('f' * len(input_data), *input_data))  # Data

        in_path = "{}_{}_input.data".format(os.path.splitext(input_file_name)[0],
                                            os.path.splitext(os.path.basename(onnx_model))[0])

        in_path = os.path.join(onnx_file_path, in_path)

        print("Saving input to {}".format(in_path))
        with open(in_path, "wb") as f:
            for packed_struct in packed_input:
                f.write(packed_struct)

        input_data = input_data.reshape(input_shape)

    elif file_type == 'image/png':
        input_data = None
        print("ERROR: File type {} currently not supported.".format(file_type))
        exit(1)

    elif file_type == 'image/jpeg':
        input_data = imageio.imread(input_file)
        input_data = np.array(input_data, dtype=float)
        input_data = input_data.astype(np.float32)

        packed_input.append(struct.pack('{}s'.format(len(magic_input)), magic_input))
        packed_input.append(struct.pack('i', input_shape[1]))  # Number of channels
        packed_input.append(struct.pack('i', input_shape[2]))  # Channel height
        packed_input.append(struct.pack('i', input_shape[3]))  # Channel width

        input_data = input_data.reshape(input_shape)

        for channel in input_data[0]:
            for row in channel:
                packed_input.append(struct.pack('f' * len(row), *row))  # Data

        in_path = "{}_{}_input.data".format(os.path.splitext(input_file_name)[0],
                                            os.path.splitext(os.path.basename(onnx_model))[0])

        in_path = os.path.join(onnx_file_path, in_path)

        print("Saving input to {}".format(in_path))
        with open(in_path, "wb") as f:
            for packed_struct in packed_input:
                f.write(packed_struct)

    elif file_type == 'image/x-portable-greymap':
        input_data = imageio.imread(input_file)
        input_data = np.array(input_data, dtype=float)
        input_data = input_data / 255.0
        input_data = input_data.astype(np.float32)

        packed_input.append(struct.pack('{}s'.format(len(magic_input)), magic_input))
        packed_input.append(struct.pack('i', input_shape[1]))  # Number of channels
        packed_input.append(struct.pack('i', input_shape[2]))  # Channel height
        packed_input.append(struct.pack('i', input_shape[3]))  # Channel width

        for row in input_data:
            packed_input.append(struct.pack('f' * len(row), *row))  # Data

        in_path = "{}_{}_input.data".format(os.path.splitext(input_file_name)[0],
                                            os.path.splitext(os.path.basename(onnx_model))[0])

        in_path = os.path.join(onnx_file_path, in_path)

        print("Saving input to {}".format(in_path))
        with open(in_path, "wb") as f:
            for packed_struct in packed_input:
                f.write(packed_struct)

        input_data = input_data.reshape(input_shape)

    elif file_type == 'random':
        input_data = np.random.uniform(r[0], r[1], input_shape)
        input_data = input_data.astype(np.float32)

        if len(input_shape) == 4:
            packed_input.append(struct.pack('{}s'.format(len(magic_input)), magic_input))
            packed_input.append(struct.pack('i', input_shape[1]))  # Number of channels
            packed_input.append(struct.pack('i', input_shape[2]))  # Channel height
            packed_input.append(struct.pack('i', input_shape[2]))  # Channel width

            for channel in input_data[0]:
                for row in channel:
                    packed_input.append(struct.pack('f' * len(row), *row))  # Data

        elif len(input_shape) == 3:
            packed_input.append(struct.pack('{}s'.format(len(magic_input)), magic_input))
            packed_input.append(struct.pack('i', input_shape[1]))  # Number of channels
            packed_input.append(struct.pack('i', 1))  # Channel height
            packed_input.append(struct.pack('i', input_shape[2]))  # Channel width

            for channel in input_data[0]:
                packed_input.append(struct.pack('f' * len(channel), *channel))  # Data

        elif len(input_shape) == 2:
            packed_input.append(struct.pack('{}s'.format(len(magic_input)), magic_input))
            packed_input.append(struct.pack('i', 1))  # Number of channels
            packed_input.append(struct.pack('i', 1))  # Channel height
            packed_input.append(struct.pack('i', input_shape[1]))  # Channel width

            packed_input.append(struct.pack('f' * len(input_data[0]), *input_data[0]))  # Data

        else:
            print("ERROR: Unsupported input shape length: {}".format(input_shape))
            exit(1)

        in_path = "{}_{}_input.data".format(os.path.splitext(input_file_name)[0],
                                            os.path.splitext(os.path.basename(onnx_model))[0])

        in_path = os.path.join(onnx_file_path, in_path)

        print("Saving input to {}".format(in_path))
        with open(in_path, "wb") as f:
            for packed_struct in packed_input:
                f.write(packed_struct)

    else:
        input_data = None
        print("ERROR: Something went wrong during input data processing...")
        exit(1)

    # Run inference on input data.
    print("Running inference...")
    outputs = backend.run_model(model, input_data, device='CPU')
    output_shape = outputs[0].shape
    print("Shape of output data: {}".format(output_shape))

    # Write outputs into our custom binary format.
    packed_output.append(struct.pack('{}s'.format(len(magic_output)), magic_output))
    packed_output.append(struct.pack('i', output_shape[1]))  # Number of outputs

    for output in outputs[0]:
        packed_output.append(struct.pack('f'*len(output), *output))  # Data

    out_path = "{}_{}_output.data".format(os.path.splitext(input_file_name)[0],
                                          os.path.splitext(os.path.basename(onnx_model))[0])
    out_path = os.path.join(onnx_file_path, out_path)
    print("Saving output to {}".format(out_path))
    with open(out_path, "wb") as f:
        for packed_struct in packed_output:
            f.write(packed_struct)


if __name__ == '__main__':
    main()
