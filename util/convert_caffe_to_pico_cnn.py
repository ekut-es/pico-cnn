# converts a caffemodel file into a pico-cnn weights file

import sys
import struct
import caffe

prototxt = sys.argv[1] 
caffemodel = sys.argv[2]
weights = sys.argv[3]

net = caffe.Net(str(prototxt), str(caffemodel), caffe.TEST)

layer_names = [n for n in net._layer_names]
num_layers = len([n for n in net.layers if n.type == 'Convolution' or n.type == 'InnerProduct'])

weights_file = open(weights, "w");

# magic number
weights_file.write("FE\n");

# name
weights_file.write("pycaffe export\n");

# num layers
weights_file.write(str(num_layers));
weights_file.write("\n");

for layer_name in layer_names[:]:
    layer_index = list(net._layer_names).index(layer_name)
    if net.layers[layer_index].type == 'Convolution':

        print(layer_name)

        weights_file.write(str(layer_name));
        weights_file.write("\n");

        kernel_height = net.params[layer_name][0].data.shape[3]
        kernel_width = net.params[layer_name][0].data.shape[2]
        num_inputs = net.params[layer_name][0].data.shape[1]
        num_outputs = net.params[layer_name][0].data.shape[0]
        num_kernels = num_inputs*num_outputs

        # kernel height
        weights_file.write(str(kernel_height))
        weights_file.write("\n")

        # kernel width
        weights_file.write(str(kernel_width))
        weights_file.write("\n")

        # number of kernels
        weights_file.write(str(num_kernels))
        weights_file.write("\n")

        # kernel weights
        for data in net.params[layer_name][0].data:
            for kernel in data:
                for line in kernel:
                    for weight in line:
                        struct.unpack('f', struct.pack('f',weight))[0]
                        weights_file.write(str(float(weight).hex()))
                        weights_file.write("\n")
     
        num_biasses = net.params[layer_name][1].data.shape[0]

        # number of biasses
        weights_file.write(str(num_biasses))
        weights_file.write("\n")

        for weight in net.params[layer_name][1].data:
            struct.unpack('f', struct.pack('f',weight))[0]
            weights_file.write(str(float(weight).hex()))
            weights_file.write("\n")

    elif net.layers[layer_index].type == 'InnerProduct':

        print(layer_name)

        weights_file.write(str(layer_name));
        weights_file.write("\n");

        kernel_height = net.params[layer_name][0].data.shape[0]
        kernel_width = net.params[layer_name][0].data.shape[1]

        # kernel height
        weights_file.write(str(kernel_height))
        weights_file.write("\n")

        # kernel width
        weights_file.write(str(kernel_width))
        weights_file.write("\n")

        # number of kernels
        weights_file.write("1\n")

        # kernel weights
        for data in net.params[layer_name][0].data:
            for weight in data:
                struct.unpack('f', struct.pack('f',weight))[0]
                weights_file.write(str(float(weight).hex()))
                weights_file.write("\n")

        num_biasses = net.params[layer_name][0].data.shape[1]

        # number of biasses
        weights_file.write(str(num_biasses))
        weights_file.write("\n")

        # bias weights
        for weight in net.params[layer_name][1].data:
            struct.unpack('f', struct.pack('f',weight))[0]
            weights_file.write(str(float(weight).hex()))
            weights_file.write("\n")
