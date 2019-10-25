import argparse
from typing import Text
import struct
import os

import numpy as np
import onnx
import caffe2.python.onnx.backend as backend

import magic
from scipy.io.wavfile import read
# from PIL import Image
import imageio
supported_file_types = ['audio/x-wav', 'image/jpeg']


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
        required=True,
        help="Path to the data the network should process."
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs='+',
        required=True,
        help="Required input shape. Format \"--shape 1 1 16000 1\" for (1, 1, 16000, 1) with "
             "(batch_size, num_input_ch, width, height)."
    )
    args = parser.parse_args()

    input_file = args.file
    input_file_path = os.path.dirname(input_file)
    input_file_name = os.path.basename(input_file)
    onnx_model = args.model
    input_shape = tuple(args.shape)

    # Check file type
    file_type = magic.from_file(input_file, mime=True)
    if file_type not in supported_file_types:
        print("Unsupported file type: {}. At the moment this tool only supports: {}".format(file_type, supported_file_types))
        exit(1)

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
        packed_input.append(struct.pack('i', input_shape[2]))  # Channel width
        packed_input.append(struct.pack('i', 1))  # Channel height
        packed_input.append(struct.pack('f' * len(input_data), *input_data))  # Data

        # tmp = os.path.splitext(os.path.basename(onnx_model))
        # tmp2 = os.path.splitext(input_file)

        in_path = "{}_{}_input.data".format(os.path.splitext(input_file)[0], os.path.splitext(os.path.basename(onnx_model))[0])
        print("Saving input to {}".format(in_path))
        with open(in_path, "wb") as f:
            for packed_struct in packed_input:
                f.write(packed_struct)

        input_data = input_data.reshape(input_shape)

    elif file_type == 'image/png':
        pass
    elif file_type == 'image/jpeg':
        pass
    else:
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

    out_path = "{}_{}_output.data".format(os.path.splitext(input_file)[0], os.path.splitext(os.path.basename(onnx_model))[0])
    print("Saving output to {}".format(out_path))
    with open(out_path, "wb") as f:
        for packed_struct in packed_output:
            f.write(packed_struct)


if __name__ == '__main__':
    main()
