from compute_graph import *
from constprop import constant_propagation
from utils import reduce_mult
from memory_manager import MemoryManager
from ir import OperationRegistry
from pico_cnn_layers import *
import cffi

import onnx.backend.base as backend_base
import onnx

import os


class BackendRep(backend_base.BackendRep):
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model 
        self.network_code = ""
        self.network_header = ""
        self.parameter_code = ""
        self.parameter_header = ""
        self.network_def = ""
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

                    graph.nodes.remove(node)
                    
                    for node in graph.nodes:
                        for num, input in enumerate(node.inputs):
                            if input == output:
                                print("node", node.name, "replacing input",
                                      input, "with", removed_input)
                                node.inputs[num] = removed_input

    def _generate_parameters(self, graph, memory_manager):
        # Generate Node Parameters
        parameter_header = "#ifndef NETWORK_PARAMETERS_H\n";
        parameter_header += "#define NETWORK_PARAMETERS_H\n";
        parameter_header += "#include \"pico-cnn/parameters.h\"\n\n"
        parameter_code = "#include \"network_parameters.h\"\n\n";
        for node in graph.nodes:
            for num, input in  enumerate(node.input_tensors):
                buffer = memory_manager.get_buffer(graph, input)
                data = node.input_tensors[input]
                
                if node.op_type == "Gemm":
                   data = data.transpose()

                type_code = "fp_t " + buffer.name + "[]"      
                declaration = "// " + str(data.shape) + "\n"
                declaration += "extern " + type_code + ";"
                definition  = type_code + " = {" + ",".join((str(x) for x in data.flatten())) + "};"
     
                parameter_code += definition + "\n\n"
                parameter_header += declaration + "\n\n"
                
        parameter_header += "#endif \n"

        self.parameter_header = parameter_header
        self.parameter_code = parameter_code

    def _select_implementations(self, graph, memory_manager):
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
        # This is not a real scheduler, for now, just assume
        # the onnx defines a valid schedule
        
        SchedulerTask = namedtuple("SchedulerTask", ["time", "node", "implementation"])
        schedule = []
        for num, node in enumerate(graph.nodes):
            schedule.append(SchedulerTask(num, node, implementations[node]))

        return schedule

    def _allocate_memory(self, graph, schedule):
        # Calculate Live Ranges

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
        graph = ComputeGraph.from_onnx(self.onnx_model.graph)
     
        print("Running constant propagation")
        constant_states = constant_propagation(graph)

        self._remove_constants(graph, constant_states)
        self._remove_nops(graph, constant_states)
         
        # Add shape information form constant propagation:
        for var, res in constant_states.items():
            if var in graph.shape_dict:
                shape = graph.shape_dict[var]
                if res.shape != shape:
                    print("Warning: Shapes do not match: ", var, res.shape, shape)
                    if res.shape is not None:
                        graph.shape_dict[var] = res.shape
            elif res.shape is not None:
                graph.shape_dict[var] = res.shape

        print("Inference graph:")
        for node in graph.nodes:
            inputs = node.inputs
            input_shapes = (str(graph.shape_dict[i]) for i in node.inputs if i in graph.shape_dict)
            outputs = node.outputs
            output_shapes = (str(graph.shape_dict[o]) for o in node.outputs if o in graph.shape_dict)
            print("{:<24}  {:<20}  {:<20}  {:<30}  {:<20}  {:<30}".format(node.name,
                                                  node.op_type,
                                                  ",".join(inputs),
                                                  ",".join(input_shapes),
                                                  ",".join(outputs),
                                                  ",".join(output_shapes)))

        memory_manager = MemoryManager()

        self._generate_parameters(graph, memory_manager)
        
        implementations = self._select_implementations(graph, memory_manager)
        schedule = self._get_schedule(graph, implementations)
        alllocation = self._allocate_memory(graph, schedule)
        

        input_names = ["input"+str(name) for name, type, shape in graph.inputs]
        output_names = ["output"+str(name) for name, type, shape in graph.outputs]
     
        input_defs = ["float *"+n for n in input_names];
        output_defs = ["float *"+n for n in output_names];
        network_def = "void network(" + ", ".join(input_defs) + ", " + ", ".join(output_defs) +  ")"

        self.network_def = network_def + ";"
        
        network_header =  "#ifndef NETWORK_H\n"
        network_header += "#define NETWORK_H\n"
        network_header += "#include \"pico-cnn/parameters.h\"\n\n"
        network_header += network_def + ";\n"
        network_header += "#endif //NETWORK_H\n"

        
        network_code : Text =  "#include \"network.h\"\n"
        network_code += "#include \"network_parameters.h\"\n\n"
        network_code += "#include \"pico-cnn/pico-cnn.h\"\n\n"
        network_code += network_def+"{\n"
     
        implementation_code = ""
        buffer_code = ""
        buffer_code_end = ""
        
        for task in schedule:
            num, node, impl = task 
            implementation_code += "  //Layer " + str(num) + " " +  node.name + " " +   node.op_type + "\n"
            implementation_code += "  //Attributes\n"
            for key, val in node.attrs.items():
                implementation_code += "  //  " + str(key) + ": " + str(val) + "\n"
            implementation_code += "  //Parameters\n"
            implementation_code += "  //Inputs: "+ ",".join(node.inputs) + "\n"
            implementation_code += "  //Outputs: "+ ",".join(node.outputs) + "\n"

            
            print(impl)
            if impl:
                implementation_code += impl.generate_code()
                implementation_code += "\n"
            
     

        for id, buffer in memory_manager.buffers.items():
            if graph.is_tensor(id):
                continue
            if graph.is_input(id):
                continue
            if graph.is_output(id):
                continue

            buffer_code += "// " + str(buffer.shape) + "\n"
            buffer_code += "  " + buffer.static_decl
            buffer_code += "\n"
                
                
        buffer_code += buffer_code_end
    
        
        network_code += buffer_code
        network_code += implementation_code
            
        network_code += "}\n"

        self.network_code = network_code
        self.network_header = network_header

        self.save(".")

    def save(self, folder):
        with open(os.path.join(folder, "network.c"), "w") as f:
            f.write(self.network_code)
     
        with open(os.path.join(folder, "network.h"), "w") as f:
            f.write(self.network_header)

        with open(os.path.join(folder, "network_parameters.h"), "w") as f:
            f.write(self.parameter_header)
     
        with open(os.path.join(folder, "network_parameters.c"), "w") as f:
            f.write(self.parameter_code)
     
            

class Backend(object):
    @classmethod
    def prepare(cls,
                model,  # type: ModelProto
                device='CPU',  # type: Text
                **kwargs  # type: Any
                ):  # type: (...) -> Optional[BackendRep]
        # TODO Remove Optional from return type
        onnx.checker.check_model(model)

        rep = BackendRep(model)
        
        return  rep
        
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
    
