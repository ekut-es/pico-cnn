from utils import reduce_mult
import numpy as np


class Buffer(object):
    def __init__(self, id, name, shape, size, dt, dtsize, alignment, is_managed):
        self.id = id
        self.name = name
        self.shape = shape
        self.size = size
        self.dt = dt
        self.typed_size = size // dtsize
        self.dtsize = dtsize
        self.alignment = alignment
        self.is_managed = 0
        self.offset = 0

    @property
    def start_ptr(self):
        if self.offset != 0:
            return "((float*)("+self.name+"+"+str(self.offset)+"))"
        
        return "((float*)"+self.name+")"

    @property
    def static_decl(self):
        return "static float " + self.name + "[" + str(self.typed_size) + "]" + ";"
    
    @classmethod
    def get(cls, graph, id : str, name="", alignment=None):
        buffer_name = "buffer"
        is_managed = True
        if graph.is_input(id):
            buffer_name = "input"
            is_managed = False
        elif graph.is_output(id):
            buffer_name = "output"
            is_managed = False
            
        if name != "":
            buffer_name = name
            is_managed = False
            
        buffer_name += id        
        shape = graph.get_shape(id)

        #TODO infer datatypes
        dt = np.float
        dtsize = 4

        size = reduce_mult(shape)*dtsize
        if alignment is None:
            alignment = dtsize
        
        
        return cls(id, buffer_name, shape, size,
                   dt, dtsize, alignment, is_managed)

class MemoryManager():
    def __init__(self):
        self.max_memory = 0
        self.free_list = []
        self.buffers = {}
        self.current_allocation_id = 1

    def allocate_memory(self, buffer_size):
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
        if id in self.buffers:
            return self.buffers[id]
        
        buffer = Buffer.get(graph, id)
        self.buffers[id] = buffer

        return self.buffers[id]
