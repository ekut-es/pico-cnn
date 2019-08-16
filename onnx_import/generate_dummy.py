from jinja2 import Environment, FileSystemLoader, Template
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "code_templates")

template_env = Environment(loader=FileSystemLoader(template_dir))


def generate_dummy_main(graph):
    template = template_env.get_template("dummy_input.c")

    attributes = {}

    inputs = graph.inputs
    outputs = graph.outputs

    if len(inputs) > 1:
        print("ERROR: Multiple inputs not supported!")
        return
    else:
        input_shape = inputs[0].shape

        if len(input_shape) == 4:
            if input_shape[0] != 1:
                print("ERROR: Inference for batch_size > 1 currently not supported!")
                return

            num_input_channels = input_shape[1]

            assert input_shape[2] == input_shape[3]
            input_channel_width = input_shape[2]
            input_channel_height = input_shape[3]

        elif len(input_shape) == 2:
            num_input_channels = 1
            input_channel_width = input_shape[0]
            input_channel_height = input_shape[1]

        else:
            print("ERROR: Creation of dummy input not supported for this input shape: {}".format(input_shape))
            return

    if len(outputs) > 1:
        print("ERROR: Multiple outputs not supported!")
        return
    else:
        output_shape = outputs[0].shape

        if len(output_shape) != 2:
            print("ERROR: Unsupported output shape: {}".format(output_shape))
            return
        else:
            output_size = output_shape[1]

    attributes["input_shape_len"] = len(input_shape)
    attributes["num_input_channels"] = num_input_channels
    attributes["input_channel_width"] = input_channel_width
    attributes["input_channel_height"] = input_channel_height
    attributes["output_size"] = output_size

    return template.render(**attributes)
