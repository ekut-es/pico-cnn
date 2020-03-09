from utils import reduce_mult
import numpy as np

__author__ = "Christoph Gerum, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


class Buffer(object):
    """
    Contains all necessary information about an input/output buffer of a CNN operation/layer.
    """
    def __init__(self, id, name, shape, size, dt, dtsize, alignment,
                 is_managed, dt_string=None, buffer_depth=None):
        self.id = id
        self._name = name
        self.shape = shape
        self.size = size
        self.dt = dt
        self.dt_string = dt_string
        self.typed_size = size // dtsize
        self.dtsize = dtsize
        self.alignment = alignment
        self.is_managed = 0
        self.offset = 0
        self.buffer_depth = buffer_depth

    @property
    def name(self):
        """
        The property 'name' was needed to strip self._name from characters that mess up the generated variable names.
        :return: The name of the buffer without '.' (if this character is in the name)
        """
        self._name = self._name.replace('.', '_')
        self._name = self._name.replace('/', '_')
        self._name = self._name.replace(':', '_')

        return self._name

    @name.setter
    def name(self, value):
        """
        Setter of the name property.
        :param value: self._name will be set to this value.
        :return:
        """
        self._name = value

    @property
    def start_ptr(self):
        if self.offset != 0:
            return "((float*)("+self.name+"+"+str(self.offset)+"))"  # TODO make dependent on dt

        return "((float*)"+self.name+")"

    @property
    def static_decl(self):
        return "static float " + self.name + "[" + str(self.typed_size) + "]" + ";"

    # @property
    # def dynamic_decl(self):  # TODO Can be removed probably
    #     return "float **" + self.name + ";"

    @classmethod
    def get(cls, graph, id: str, name="", alignment=None, dt_string=None):
        """
        Return a Buffer object containing the information passed to this method.
        :param graph: ComputeGraph representing the CNN
        :param id: Unique identifier of the buffer
        :param name: Name of the buffer, can be omitted
        :param alignment: Alignment of the data type in memory. If omitted alignment=4 is assumed.
        :param dt_string: Field to support more data types in the future.
        :return: Buffer object containing the information passed to this method.
        """
        buffer_name = "buffer_"
        is_managed = True
        if graph.is_input(id):
            buffer_name = "input_"
            is_managed = False
        elif graph.is_output(id):
            buffer_name = "output_"
            is_managed = False
            
        if name != "":
            buffer_name = name
            is_managed = False
            
        buffer_name += id        
        shape = graph.get_shape(id)

        # TODO: Refactor this because buffer_depth is a terrible name that does not represent what it should.
        # This variable is about whether we have multiple channels or not
        if len(shape) == 1 or len(shape) == 2:
            buffer_depth = 1
        elif len(shape) == 4:
            buffer_depth = 2
        elif len(shape) == 3:
            buffer_depth = 2  # TODO: When do we actually need depth = 3???
        else:
            buffer_depth = 0

        # TODO infer data types as soon as we support more data types (e.g. fixed-point)
        dt = np.float
        dtsize = 4

        size = reduce_mult(shape)*dtsize
        if alignment is None:
            alignment = dtsize

        return cls(id, buffer_name, shape, size,
                   dt, dtsize, alignment, is_managed, dt_string, buffer_depth)


class MemoryManager:
    """
    Containing the buffer objects associated with the inputs and outputs of all layers in the ComputeGraph.
    """
    def __init__(self):
        self.max_memory = 0
        self.free_list = []
        self.buffers = {}
        self.current_allocation_id = 1

    def allocate_memory(self, buffer_size):
        """
        Not used at the moment
        :param buffer_size:
        :return:
        """
        print("allocatiing", buffer_size)
        print(self.free_list)
        for block in self.free_list:
            if buffer_size <= block["size"]:
                if block["size"] != buffer_size:
                    new_block = dict(id = self.current_allocation_id,
                                     start = copy(block["start"]),
                                     size = buffer_size)
                    block["size"] -= buffer_size
                    block["start"] += buffer_size
                    print("block:", block)
                    print("new_block: ", new_block)
                    self.current_allocation_id += 1
                    return new_block
                else:
                    return block

        new_block = dict(id = self.current_allocation_id,
                         start = self.max_memory,
                         size = buffer_size)
        self.max_memory += buffer_size
        self.current_allocation_id += 1

        print("freshly allocated block", new_block)
        
        return new_block

    def free_memory(self, buffer):
        """
        Not used at the moment
        :param buffer:
        :return:
        """
        print("deallocating", buffer)
        def find_buffer(start, size):
            end = start + size
            for buffer in self.free_list:
                buffer_start = buffer["start"]
                buffer_end = buffer["start"] + buffer["size"]

                max_start = max(start, buffer_start)
                min_end = min(end, buffer_end)

                if max_start <= min_end:
                    return buffer

            return None

        overlapping_buffer = find_buffer(buffer ["start"], buffer["size"])
        while(overlapping_buffer):
            self.free_list.remove(overlapping_buffer)

            start = min(buffer["start"], overlapping_buffer["start"])
            end = max(buffer["start"]+buffer["size"], overlapping_buffer["start"]+overlapping_buffer["size"])

            buffer["start"] = start
            buffer["size"] = end - start

            overlapping_buffer = find_buffer(buffer["start"], buffer["size"])
            
        self.free_list.append(buffer)
        self.free_list.sort(key = lambda x : x["start"])

    def allocate(self, schedule):
        return
        
    def get_buffer(self, graph, id):
        """
        If 'id' is already in self.buffers return the Buffer object.
        If not create a Buffer object and place it in self.buffers and return it.
        :param graph: ComputeGraph representing the CNN
        :param id: Identifier of the buffer
        :return: Buffer object
        """
        if id in self.buffers:
            return self.buffers[id]
        
        buffer = Buffer.get(graph, id)
        self.buffers[id] = buffer

        return self.buffers[id]
