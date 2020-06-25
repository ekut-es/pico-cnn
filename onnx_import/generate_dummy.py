from jinja2 import Environment, FileSystemLoader
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "code_templates")

template_env = Environment(loader=FileSystemLoader(template_dir))

__author__ = "Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


def generate_dummy_main(graph):
    """
    Generate code that creates random input values and calls the CNN.
    :param graph: ComputeGraph representing the CNN.
    :return: String containing the generated code.
    """
    template = template_env.get_template("dummy_input.c")

    attributes = {}

    inputs = graph.inputs
    outputs = graph.outputs

    num_input_dims = 0
    num_input_batches = 0
    num_input_channels = 0
    input_channel_height = 0
    input_channel_width = 0

    if len(inputs) > 1:
        print("ERROR: Multiple inputs not supported!")
        exit(1)
    else:
        input_shape = graph.shape_dict[inputs[0].name]

        if len(input_shape) == 4:
            if input_shape[0] != 1:
                print("ERROR: Inference for batch_size > 1 currently not supported!")
                exit(1)

            num_input_dims = 4
            num_input_batches = input_shape[0]
            num_input_channels = input_shape[1]
            input_channel_height = input_shape[2]
            input_channel_width = input_shape[3]

        elif len(input_shape) == 2:
            num_input_dims = 2
            num_input_channels = 1
            input_channel_height = input_shape[0]
            input_channel_width = input_shape[1]

        elif len(input_shape) == 3:
            if input_shape[0] != 1:
                print("ERROR: Inference for batch_size > 1 currently not supported!")
                exit(1)

            num_input_dims = 3
            num_input_batches = input_shape[0]
            num_input_channels = input_shape[1]
            input_channel_height = 1
            input_channel_width = input_shape[2]

        else:
            print("ERROR: Creation of dummy input not supported for this input shape: {}".format(input_shape))
            exit(1)

    num_output_dims = 0
    num_output_batches = 0
    num_output_channels = 0
    output_channel_height = 0
    output_channel_width = 0

    if len(outputs) > 1:
        print("ERROR: Multiple outputs not supported!")
        exit(1)
    else:
        output_shape = graph.shape_dict[outputs[0].name]

        if len(output_shape) == 2:
            num_output_dims = 2
            num_output_batches = output_shape[0]
            num_output_channels = output_shape[1]
            output_channel_height = 0
            output_channel_width = 0
        elif len(output_shape) == 4:
            assert output_shape[0] == 1
            num_output_dims = 4
            num_output_batches = output_shape[0]
            num_output_channels = output_shape[1]
            output_channel_height = output_shape[2]
            output_channel_width = output_shape[3]
        else:
            print("ERROR: Unsupported output shape: {}".format(output_shape))
            exit(1)

    attributes["num_input_dims"] = num_input_dims
    attributes["num_input_batches"] = num_input_batches
    attributes["num_input_channels"] = num_input_channels
    attributes["input_channel_height"] = input_channel_height
    attributes["input_channel_width"] = input_channel_width

    attributes["num_output_dims"] = num_output_dims
    attributes["num_output_batches"] = num_output_batches
    attributes["num_output_channels"] = num_output_channels
    attributes["output_channel_height"] = output_channel_height
    attributes["output_channel_width"] = output_channel_width

    return template.render(**attributes)


def generate_reference_main(graph):
    """
    Generate code that creates random input values and calls the CNN.
    :param graph: ComputeGraph representing the CNN.
    :return: String containing the generated code.
    """
    template = template_env.get_template("reference_input.c")

    attributes = {}

    inputs = graph.inputs
    outputs = graph.outputs

    num_input_dims = 0
    num_input_batches = 0
    num_input_channels = 0
    input_channel_height = 0
    input_channel_width = 0

    if len(inputs) > 1:
        print("ERROR: Multiple inputs not supported!")
        exit(1)
    else:
        input_shape = inputs[0].shape

        if len(input_shape) == 4:
            if input_shape[0] != 1:
                print("ERROR: Inference for batch_size > 1 currently not supported!")
                exit(1)

            num_input_dims = 4
            num_input_batches = input_shape[0]
            num_input_channels = input_shape[1]
            input_channel_height = input_shape[2]
            input_channel_width = input_shape[3]

        elif len(input_shape) == 2:
            num_input_dims = 2
            num_input_channels = 1
            input_channel_height = input_shape[0]
            input_channel_width = input_shape[1]

        elif len(input_shape) == 3:
            if input_shape[0] != 1:
                print("ERROR: Inference for batch_size > 1 currently not supported!")
                exit(1)

            num_input_dims = 3
            num_input_batches = input_shape[0]
            num_input_channels = input_shape[1]
            input_channel_height = 1
            input_channel_width = input_shape[2]

        else:
            print("ERROR: Creation of dummy input not supported for this input shape: {}".format(input_shape))
            exit(1)

    num_output_dims = 0
    num_output_batches = 0
    num_output_channels = 0
    output_channel_height = 0
    output_channel_width = 0

    if len(outputs) > 1:
        print("ERROR: Multiple outputs not supported!")
        exit(1)
    else:
        output_shape = graph.shape_dict[outputs[0].name]

        if len(output_shape) == 2:
            num_output_dims = 2
            num_output_batches = output_shape[0]
            num_output_channels = output_shape[1]
            output_channel_height = 0
            output_channel_width = 0
        elif len(output_shape) == 4:
            assert output_shape[0] == 1
            num_output_dims = 4
            num_output_batches = output_shape[0]
            num_output_channels = output_shape[1]
            output_channel_height = output_shape[2]
            output_channel_width = output_shape[3]
        else:
            print("ERROR: Unsupported output shape: {}".format(output_shape))
            exit(1)

    attributes["num_input_dims"] = num_input_dims
    attributes["num_input_batches"] = num_input_batches
    attributes["num_input_channels"] = num_input_channels
    attributes["input_channel_height"] = input_channel_height
    attributes["input_channel_width"] = input_channel_width

    attributes["num_output_dims"] = num_output_dims
    attributes["num_output_batches"] = num_output_batches
    attributes["num_output_channels"] = num_output_channels
    attributes["output_channel_height"] = output_channel_height
    attributes["output_channel_width"] = output_channel_width

    return template.render(**attributes)
