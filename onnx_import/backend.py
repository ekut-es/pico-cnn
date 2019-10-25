from compute_graph import *
from constprop import constant_propagation
from utils import reduce_mult
from memory_manager import MemoryManager
from ir import OperationRegistry
from ir import CodeRegistry
from pico_cnn import *
from memory_allocation import *
from generate_dummy import *
# import cffi

import onnx.backend.base as backend_base
import onnx

import os
import struct


class BackendRep(backend_base.BackendRep):
    def __init__(self, onnx_model, model_name):
        self.onnx_model = onnx_model
        self.model_name = model_name
        self.network_code = ""
        self.network_header = ""
        self.parameter_code = ""
        self.parameter_header = ""
        self.initialization_code = ""
        self.initialization_header = ""
        self.network_def = ""
        self.cleanup_header = ""
        self.cleanup_code = ""
        self.weights_file = ""
        self.packed_file = list()
        self.makefile = ""
        self.dummy_input = ""
        self.reference_input = ""
        self._export_model()

    def run(self, inputs, **kwargs):
        ffibuilder = cffi.FFI()
        print(self.network_def)
        ffibuilder.cdef(self.network_def)

        ffibuilder.set_source("__network", self.network_header,
                              sources=["network.c", "network_parameters.c"],
                              include_dirs=["/local/gerum/projects/ocean12/speech_recognition/runtime/pico-cnn"],
                              libraries=["m", "pthread"],
                              extra_compile_args=["-O3", "-g", "-std=c99"])

        ffibuilder.compile(verbose=True)

    def _remove_constants(self, graph, constant_states):
        # Remove nodes with constant values
        for node in list(graph.nodes):
            is_constant = True
            for output in node.outputs:

                if constant_states[output].value is None:
                    is_constant = False

            if is_constant:
                graph.remove_node(node)

    def _remove_nops(self, graph, constant_states):
        # Remove nop nodes
        for node in list(graph.nodes):
            if node.op_type == "Unsqueeze":
                inp = node.inputs[0]
                out = node.outputs[0]

                print("Removing nop", node.name)
                graph.nodes.remove(node)

                for node in graph.nodes:
                    for num, input in enumerate(node.inputs):
                        if input == out:
                            node.inputs[num] = inp

            if node.op_type == "Reshape":
                input_state = constant_states[node.inputs[0]]
                reshape_state = constant_states[node.inputs[1]]
                is_nop = False
                if reshape_state.value is not None:
                    for i, dim in enumerate(reshape_state.value):
                        if dim == -1 and i == len(reshape_state.value):
                            is_nop = True
                        elif i < len(input_state.shape) and dim == input_state.shape[i]:
                            is_nop = True
                        else:
                            is_nop = False
                            break

                if is_nop:
                    removed_input = node.inputs[0]
                    output = node.outputs[0]

                    print("Removing nop", node.name)
                    graph.nodes.remove(node)

                    for node in graph.nodes:
                        for num, input in enumerate(node.inputs):
                            if input == output:
                                print("node", node.name, "replacing input",
                                      input, "with", removed_input)
                                node.inputs[num] = removed_input

    def _generate_parameters(self, graph, memory_manager):
        """
        Legacy function to generate a .h and .c file containing all kernel and bias values.
        :param graph: ComputeGraph of the parsed onnx model.
        :param memory_manager: MemoryManager containing information about input and output buffers of each operation.
        :return:
        """
        # Generate Node Parameters
        parameter_header = "#ifndef NETWORK_PARAMETERS_H\n"
        parameter_header += "#define NETWORK_PARAMETERS_H\n"
        parameter_header += "#include \"pico-cnn/parameters.h\"\n\n"
        parameter_code = "#include \"network_parameters.h\"\n\n"
        for node in graph.nodes:
            for num, input in enumerate(node.input_tensors):
                buffer = memory_manager.get_buffer(graph, input)
                data = node.input_tensors[input]

                if node.op_type == "Gemm":
                   data = data.transpose()

                type_code = "fp_t " + buffer.name + "[]"
                declaration = "// " + str(data.shape) + "\n"
                declaration += "extern " + type_code + ";"
                definition = type_code + " = {" + ",".join((str(x) for x in data.flatten())) + "};"

                parameter_code += definition + "\n\n"
                parameter_header += declaration + "\n\n"

        parameter_header += "#endif \n"

        self.parameter_header = parameter_header
        self.parameter_code = parameter_code

    def _generate_weights_file(self, graph):
        """
        Generate a binary file containing all kernel and bias values.
        This generated file can then be read by the pico-cnn function in io/read_binary_weights.h
        :param graph: ComputeGraph of the parsed onnx model.
        :return:
        """
        # weights_file = "FE\n"
        # weights_file += "tiny-dnn export\n"

        packed_file = list(bytes())

        tupac = bytes("FD\n", "ascii")
        packed_file.append(struct.pack('{}s'.format(len(tupac)), tupac))
        packed_file.append(struct.pack('{}s'.format(len(self.model_name)+1), bytes(self.model_name+"\n", "ascii")))

        num_layers = 0

        for node in graph.nodes:
            if len(node.input_tensors) > 0 and node.op_type != "Reshape":
                num_layers += 1

        # weights_file += str(num_layers) + "\n"
        packed_file.append(struct.pack('i', num_layers))

        # weight_data = ""
        weights_packed = list(bytes())

        for node in graph.nodes:
            for num, input in enumerate(node.input_tensors):
                if node.op_type == "Reshape":
                    continue

                data = node.input_tensors[input]

                if node.op_type == "Gemm":
                    data = data.transpose()

                if len(data.shape) == 4:
                    # weight_data += node.name + "\n"

                    tupac = bytes(node.name+"\n", "ascii")
                    weights_packed.append(struct.pack('{}s'.format(len(tupac)), tupac))

                    height = data.shape[2]  # height
                    width = data.shape[3]  # width
                    num_data = data.shape[0] * data.shape[1]  # num_kernels

                    # weight_data += str(height) + "\n"
                    # weight_data += str(width) + "\n"
                    # weight_data += str(num_data) + "\n"

                    weights_packed.append(struct.pack('i', height))
                    weights_packed.append(struct.pack('i', width))
                    weights_packed.append(struct.pack('i', num_data))

                    for channel in data:
                        for kernel in channel:
                            for row in kernel:
                                weights_packed.append(struct.pack('f'*len(row), *row))

                elif len(data.shape) == 3:
                    tupac = bytes(node.name + "\n", "ascii")
                    weights_packed.append(struct.pack('{}s'.format(len(tupac)), tupac))

                    height = 1
                    width = data.shape[2]
                    num_data = data.shape[0] * data.shape[1]

                    # weight_data += str(height) + "\n"
                    # weight_data += str(width) + "\n"
                    # weight_data += str(num_data) + "\n"

                    weights_packed.append(struct.pack('i', height))
                    weights_packed.append(struct.pack('i', width))
                    weights_packed.append(struct.pack('i', num_data))

                    for channel in data:
                        for kernel in channel:
                            weights_packed.append(struct.pack('f'*len(kernel), *kernel))

                elif len(data.shape) == 2:
                    # weight_data += node.name + "\n"

                    tupac = bytes(node.name + "\n", "ascii")
                    weights_packed.append(struct.pack('{}s'.format(len(tupac)), tupac))

                    height = data.shape[0]  # height
                    width = data.shape[1]  # width
                    num_data = 1  # num_kernels

                    # weight_data += str(height) + "\n"
                    # weight_data += str(width) + "\n"
                    # weight_data += str(num_data) + "\n"

                    weights_packed.append(struct.pack('i', height))
                    weights_packed.append(struct.pack('i', width))
                    weights_packed.append(struct.pack('i', num_data))

                    for row in data:
                        weights_packed.append(struct.pack('f'*len(row), *row))

                elif len(data.shape) == 1:
                    num_data = data.shape[0]  # num_biases

                    # weight_data += str(num_data) + "\n"

                    weights_packed.append(struct.pack('i', num_data))
                    weights_packed.append(struct.pack('f'*len(data), *data))

                else:
                    print("ERROR: Unknown input tensor shape!")
                    exit(1)
                    # weight_data = ""

                # This handles the case that no bias values are available in the onnx file.
                # So we need to add num_biases = 0 into the binary file.
                if len(node.input_tensors) == 1:
                    print("No biases in onnx file.")
                    weights_packed.append(struct.pack('i', 0))

                # temp = "\n".join((str(float(x).hex()) for x in data.flatten()))
                # weight_data += temp + "\n"

        # weights_file += weight_data
        packed_file += weights_packed

        # self.weights_file = weights_file
        self.packed_file = packed_file

    def _generate_network_initialization(self, graph, memory_manager):
        """
        Generate code that allocates all necessary input and output buffers of all operations.
        At the moment everything will be written to a single .h file. This should be changed in the future.
        :param graph: ComputeGraph of the parsed onnx model.
        :param memory_manager: MemoryManager containing information about input and output buffers of each operation.
        :return:
        """
        initialization_header = "#ifndef NETWORK_INITIALIZATION_H\n"
        initialization_header += "#define NETWORK_INITIALIZATION_H\n"
        initialization_header += "#include <stdlib.h>\n"
        initialization_header += "#include \"pico-cnn/parameters.h\"\n\n"
        initialization_header += "void initialize_network();\n\n"
        # initialization_header += "void initialize();\n\n"
        initialization_header += "fp_t*** kernels;\n"
        initialization_header += "fp_t** biases;\n"

        initialization_code = "#include \"network_initialization.h\"\n\n"
        initialization_code += "void initialize_network() {\n\n"

        num_layers = 0

        for node in graph.nodes:
            """Do not count the reshape layers as the input tensor will only define the dimensions"""
            if len(node.input_tensors) > 0 and node.op_type != "Reshape":
                num_layers += 1

        """The arrays kernels and biases will be used to pass only two variables to read_binary_weights"""
        initialization_code += "kernels = (fp_t***) malloc({} * sizeof(fp_t**));\n".format(num_layers)
        initialization_code += "biases = (fp_t**) malloc({} * sizeof(fp_t*));\n\n".format(num_layers)

        pos = -1

        """Iterate over all nodes in the graph and generate the corresponding allocation code."""
        for node in graph.nodes:

            if len(node.input_tensors) > 0 and node.op_type != "Reshape":
                pos += 1

            initialization_header += "// Layer: " + node.name + ", Operation: " + node.op_type + "\n"
            initialization_code += "// Layer: " + node.name + ", Operation: " + node.op_type + "\n"

            # Allocate memory for kernels and biases
            initialization_header += "// Inputs\n"
            initialization_code += "// Inputs\n"
            for num, input in enumerate(node.input_tensors):

                if node.op_type == "Reshape":
                    continue

                buffer = memory_manager.get_buffer(graph, input)
                data = node.input_tensors[input]

                # if len(data.shape) == 4:
                #     num_kernels = data.shape[0] * data.shape[1]
                #     kernel_size = data.shape[2] * data.shape[3]
                # elif len(data.shape) == 1:
                #     num_kernels = data.shape[0]
                #     kernel_size = 1

                initialization_header += "// " + str(buffer.shape) + "\n"
                data_type = "fp_t "
                for i in range(buffer.buffer_depth):
                    data_type += "*"

                initialization_header += data_type + buffer.name + ";\n"

                initialization_code += "// " + str(buffer.shape) + ""  # TODO maybe we sometimes need \n

                functionality = CodeRegistry.get_funct("KernelAllocation")
                impl = functionality[0].create(buffer, pos)

                if impl:
                    initialization_code += impl.generate_code()
                    initialization_code += "\n"

            initialization_header += "// Outputs\n"
            initialization_code += "// Outputs\n"
            for num, output in enumerate(node.outputs):
                buffer = memory_manager.get_buffer(graph, output)

                initialization_header += "// " + str(buffer.shape) + "\n"

                data_type = "fp_t "
                for i in range(buffer.buffer_depth):
                    data_type += "*"

                initialization_header += data_type + buffer.name + ";\n"

                initialization_code += "// " + str(buffer.shape) + ""  # TODO maybe we sometimes need \n

                functionality = CodeRegistry.get_funct("OutputAllocation")
                impl = functionality[0].create(buffer)

                if impl:
                    initialization_code += impl.generate_code()
                    initialization_code += "\n"

            initialization_header += "\n\n"
            initialization_code += "\n\n"

        initialization_code += "}\n"

        #initialization_header += initialization_code  # TODO Everything to the .h file???
        initialization_header += "#endif //NETWORK_INITIALIZATION_H\n"

        self.initialization_header = initialization_header
        self.initialization_code = initialization_code

    def _generate_network_cleanup(self, graph, memory_manager):
        """
        Generate code that frees all previously allocated buffer memory of all operations.
        :param graph: ComputeGraph of the parsed onnx model.
        :param memory_manager: MemoryManager containing information about input and output buffers of each operation.
        :return:
        """
        cleanup_header = "#ifndef NETWORK_CLEANUP_H\n"
        cleanup_header += "#define NETWORK_CLEANUP_H\n"
        cleanup_header += "#include <stdlib.h>\n"
        cleanup_header += "#include \"pico-cnn/parameters.h\"\n"
        cleanup_header += "#include \"network_initialization.h\" \n\n"
        cleanup_header += "void cleanup_network(); \n\n"
        cleanup_header += "#endif //NETWORK_CLEANUP_H\n"

        # cleanup_header += "void cleanup();\n\n"

        cleanup_code = "#include \"network_cleanup.h\"\n\n"
        #cleanup_code += "#include \"network_initialization.h\"\n\n"
        cleanup_code += "void cleanup_network() {\n"

        for num, buffer_id in enumerate(memory_manager.buffers):
            buffer = memory_manager.get_buffer(graph, buffer_id)

            functionality = CodeRegistry.get_funct("BufferCleanup")
            impl = functionality[0].create(buffer)

            if impl:
                cleanup_code += impl.generate_code()
                cleanup_code += "\n"

        cleanup_code += "\nfree(kernels);\nfree(biases);\n"

        cleanup_code += "}\n"

        #cleanup_header += cleanup_code


        self.cleanup_header = cleanup_header
        self.cleanup_code = cleanup_code

    def _select_implementations(self, graph, memory_manager):
        """
        Function to select the first of possibly multiple implementation candidates
        for a each operation in the ComputeGraph.
        TODO: In the future this function will be extended to select different implementations (naive/armPerfLibs/openMP)
        :param graph: ComputeGraph of the parsed onnx model.
        :param memory_manager: MemoryManager containing information about input and output buffers of each operation.
        :return: Dictionary containing implementations of all the nodes in the ComputeGraph
        """
        implementations = {}
        for node in graph.nodes:
            choices = []
            for op in OperationRegistry.get_ops(node.op_type):
                candidate = op.create(node, graph, memory_manager)
                if candidate is not None:
                    choices.append(candidate)

            if len(choices) >= 1:
                implementations[node] = choices[0]
            else:
                implementations[node] = None

        return implementations

    def _get_schedule(self, graph, implementations):
        """
        This is not a real scheduler, for now, just assume the onnx defines a valid schedule.
        The functions just enumerates all implementations and returns a list.
        :param graph: ComputeGraph of the parsed onnx model.
        :param implementations: Dictionary containing the previously selected implementations of all operations in the ComputeGraph.
        :return: List of named tuples ("SchedulerTask", ["time", "node", "implementation"])
        """
        SchedulerTask = namedtuple("SchedulerTask", ["time", "node", "implementation"])
        schedule = []
        for num, node in enumerate(graph.nodes):
            schedule.append(SchedulerTask(num, node, implementations[node]))

        return schedule

    def _allocate_memory(self, schedule):
        """
        Calculate Live Ranges and print them. For debug purposes.
        :param schedule: Previously comuted pseudo-schedule.
        :return:
        """
        range_starts = {}
        range_ends = {}

        for num, node, impl in schedule:
            for output in node.outputs:
                range_starts[output] = num
            for input in node.inputs:
                if input in range_starts:
                    range_ends[input] = num

        print("Live Ranges:")
        for name in sorted(range_starts.keys()):
            print("{:^5}".format(name),  end="")
        print()

        for num, _, _ in schedule:
            for name in sorted(range_starts.keys()):
                if num < range_starts[name]:
                    print("     ", end="")
                elif num == range_starts[name]:
                    print("  s  ", end="")
                elif num < range_ends[name]:
                    print("  |  ", end="")
                elif num == range_ends[name]:
                    print("  e  ", end="")
                else:
                    print("     ", end="")
            print()

    def _export_model(self):
        """
        Parses the onnx model to be represented as a ComputeGraph and then calls all
        functions needed for generating the pico-cnn code.
        :return:
        """
        graph = ComputeGraph.from_onnx(self.onnx_model.graph)

        print("Running constant propagation")
        constant_states = constant_propagation(graph)

        self._remove_constants(graph, constant_states)
        self._remove_nops(graph, constant_states)

        # Add shape information from constant propagation:
        for var, res in constant_states.items():
            if var in graph.shape_dict:
                shape = graph.shape_dict[var]
                if res.shape != shape:
                    print("Warning: Shapes do not match: ", var, res.shape, shape)
                    if res.shape is not None:
                        print("Replacing shape {} with {}".format(shape, res.shape))
                        graph.shape_dict[var] = res.shape
            elif res.shape is not None:
                graph.shape_dict[var] = res.shape

        print("Inference graph:")
        for node in graph.nodes:
            inputs = node.inputs
            input_shapes = (str(graph.shape_dict[i]) for i in node.inputs if i in graph.shape_dict)
            outputs = node.outputs
            output_shapes = (str(graph.shape_dict[o]) for o in node.outputs if o in graph.shape_dict)
            print("{:<24}  {:<20}  {:<30}  {:<30}  {:<20}  {:<30}".format(node.name,
                                                                          node.op_type,
                                                                          ",".join(inputs),
                                                                          ",".join(input_shapes),
                                                                          ",".join(outputs),
                                                                          ",".join(output_shapes)))

        memory_manager = MemoryManager()

        self._generate_weights_file(graph)

        self.dummy_input = generate_dummy_main(graph)

        self.reference_input = generate_reference_main(graph)

        self._generate_network_initialization(graph, memory_manager)

        self._generate_network_cleanup(graph, memory_manager)

        implementations = self._select_implementations(graph, memory_manager)
        schedule = self._get_schedule(graph, implementations)
        self._allocate_memory(schedule)

        input_names = ["input_"+name.replace('.', '') for name, type, shape in graph.inputs]
        output_names = ["output_"+name.replace('.', '') for name, type, shape in graph.outputs]

        """Currently we only allow single input (no batch processing) to the CNN, but this may be multi-channel input"""
        inputs = graph.inputs
        if len(inputs) > 1:
            print("ERROR: Multiple inputs not supported!")
            exit(1)
        else:
            input_shape = inputs[0].shape
            print("Input shape: {}".format(input_shape))

            if len(input_shape) == 4:
                if input_shape[0] != 1:
                    print("ERROR: Inference for batch_size > 1 currently not supported!")
                    exit(1)

                input_defs = ["float **"+n for n in input_names]

            elif len(input_shape) == 3:
                if input_shape[0] != 1:
                    print("ERROR: Inference for batch_size > 1 currently not supported!")
                    exit(1)

                input_defs = ["float **"+n for n in input_names]

            elif len(input_shape) == 2:
                print("Input is one-dimensional (batch_size = 1 and num_input_channels = 1)")
                input_defs = ["float *"+n for n in input_names]

        # TODO: Has to be changed as soon as be want to support multiple other data types (e.g. fixed-point)
        output_defs = ["float *"+n for n in output_names]
        network_def = "void network(" + ", ".join(input_defs) + ", " + ", ".join(output_defs) + ")"
        # network_def = "int network(" + ", ".join(input_defs) + ")"

        self.network_def = network_def + ";"

        # TODO: Separate definition and implementation in the future.
        #network_header = "#ifndef NETWORK_H\n"
        #network_header += "#define NETWORK_H\n"
        #network_header += "#include \"pico-cnn/parameters.h\"\n\n"
        # network_header += network_def + ";\n"
        # network_header += "#endif //NETWORK_H\n"

        network_code: Text = "#include \"network.h\"\n\n"
        #network_code = "#include \"network_initialization.h\"\n"
        #network_code += "#include \"network_cleanup.h\"\n\n"
        #network_code += "#include \"pico-cnn/pico-cnn.h\"\n\n"
        network_code += network_def+"{\n"

        implementation_code = ""

        """Iterate over all tasks in the schedule, put some debug info in the code and the pico-cnn implementation."""
        for task in schedule:
            num, node, impl = task
            implementation_code += "    //Layer " + str(num) + " " + node.name + " " + node.op_type + "\n"
            implementation_code += "    //Attributes\n"
            for key, val in node.attrs.items():
                implementation_code += "    //  " + str(key) + ": " + str(val) + "\n"
            implementation_code += "    //Parameters\n"
            implementation_code += "    //Inputs: " + ",".join(node.inputs) + "\n"
            implementation_code += "    //Outputs: " + ",".join(node.outputs) + "\n"

            if impl:
                implementation_code += impl.generate_code()
                implementation_code += "\n"
            else:
                print("ERROR: Unsupported layer: {}! Aborting code generation.".format(node.op_type))
                return 1

        # TODO: What does this loop do?
        for id, buffer in memory_manager.buffers.items():
            if graph.is_tensor(id):
                continue
            if graph.is_input(id):
                continue
            if graph.is_output(id):
                continue

        network_code += implementation_code

        network_code += "}\n\n"

        network_header = "#ifndef NETWORK_H\n"
        network_header += "#define NETWORK_H\n\n"
        network_header += "#include \"pico-cnn/parameters.h\"\n"
        network_header += "#include \"network_initialization.h\"\n"
        network_header += "#include \"network_cleanup.h\"\n"
        network_header += "#include \"pico-cnn/pico-cnn.h\"\n\n"
        network_header += network_def + "; \n\n"
        network_header += "#endif //NETWORK_H\n"

        self.network_code = network_code
        self.network_header = network_header

        """
        Create Makefile containing a target for the generated dummy input and a network specific one.
        The code for the network specific input has to be written manually.
        """
        # TODO: Does this need to be more sophisticated?
        self.makefile = "CC = gcc\n"
        self.makefile += "CFLAGS = -Wall -g\n"
        self.makefile += "LDFLAGS = -lm\n\n"
        self.makefile += "# path to the pico-cnn library\n"
        self.makefile += "LIBPATH = ../../../pico-cnn/lib/libpico-cnn.a\n\n"
        self.makefile += "# list of all generated .c files.\n"
        self.makefile += "#TODO: right now, all .c files are compiled in each make call: change for higher effiency?\n"
        self.makefile += "NETWORK_LIST = network_initialization.c network_cleanup.c network.c"
        self.makefile += "\n\ndummy_input: dummy_input.c $(NETWORK_LIST)\n\t"
        self.makefile += "make library --directory=../../../pico-cnn\n\t"
        self.makefile += "$(CC) dummy_input.c $(NETWORK_LIST) $(LIBPATH) -I../../.. $(CFLAGS) $(LDFLAGS) -o dummy_input"
        self.makefile += "\n\nreference_input: reference_input.c $(NETWORK_LIST) \n\t"
        self.makefile += "make library --directory=../../../pico-cnn\n\t"
        self.makefile += "$(CC) reference_input.c $(NETWORK_LIST) $(LIBPATH) -I../../.. $(CFLAGS) $(LDFLAGS) -o reference_input"
        self.makefile += "\n\n{}: {}.c $(NETWORK_LIST) \n\t".format(self.model_name, self.model_name)
        self.makefile += "make library --directory=../../../pico-cnn\n\t"
        self.makefile += "$(CC) {}.c $(NETWORK_LIST) $(LIBPATH) -I../../.. $(CFLAGS) $(LDFLAGS) -o {}".format(self.model_name, self.model_name)
        self.makefile += "\n\nall: dummy_input reference_input {}".format(self.model_name)
        self.makefile += "\n\nclean:\n\t rm -rf {} dummy_input reference_input\n".format(self.model_name)

        self.save("./generated_code/{}".format(self.model_name))

    def save(self, folder):
        """
        Save the generated code and binary weights to files in the specified directory.
        :param folder: Directory where files should be saved to.
        :return:
        """
        try:
            os.makedirs(folder)
            print("Created directory for generated code.")
        except FileExistsError:
            pass

        with open(os.path.join(folder, "network.c"), "w") as f:
             f.write(self.network_code)

        with open(os.path.join(folder, "network.h"), "w") as f:
            f.write(self.network_header)

        with open(os.path.join(folder, "network_initialization.c"), "w") as f:
            f.write(self.initialization_code)

        with open(os.path.join(folder, "network_initialization.h"), "w") as f:
            f.write(self.initialization_header)

        with open(os.path.join(folder, "network_cleanup.c"), "w") as f:
            f.write(self.cleanup_code)

        with open(os.path.join(folder, "network_cleanup.h"), "w") as f:
            f.write(self.cleanup_header)

        # with open(os.path.join(folder, "network.weights"), "w") as f:
        #     f.write(self.weights_file)

        with open(os.path.join(folder, "network.weights.bin"), "wb") as f:
            for packed_struct in self.packed_file:
                f.write(packed_struct)

        with open(os.path.join(folder, "Makefile"), "w") as f:
            f.write(self.makefile)

        with open(os.path.join(folder, "dummy_input.c"), "w") as f:
            f.write(self.dummy_input)

        with open(os.path.join(folder, "reference_input.c"), "w") as f:
            f.write(self.reference_input)


class Backend(object):
    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                model_name,  # type: Text
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[BackendRep]
        # TODO Remove Optional from return type
        onnx.checker.check_model(model)

        rep = BackendRep(model, model_name)

        return rep


def export_data(config):
    print("Exporting_input_data")
    train_set, dev_set, test_set = dataset.SpeechDataset.splits(config)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    data, label = next(iter(test_loader))
    data = data.numpy().flatten()

    data_code = "#ifndef INPUT_DATA_H\n"
    data_code += "#include \"pico-cnn/parameters.h\"\n\n"
    data_code += "fp_t input[] = {" + ",".join((str(x) for x in data)) + "};\n"
    data_code += "#endif //INPUT_DATA_H\n"
    with open("input_data.h", "w") as f:
        f.write(data_code)
