from collections import defaultdict

__author__ = "Christoph Gerum, Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


class OperationRegistry(object):
    """
    This class is used to register, store and return the operations inheriting from BaseLayer.
    """
    ops = []
    ops_by_name = defaultdict(list)
    ops_by_operator = defaultdict(list)

    @classmethod
    def register(cls, op):
        if op not in cls.ops:
            cls.ops.append(op)
            cls.ops_by_name[op.name].append(op)
            cls.ops_by_operator[op.operator].append(op)

    @classmethod
    def get_ops(cls, operation):
        return cls.ops_by_operator[operation]


class CodeRegistry(object):
    """
    This class is used to register, store and return the memory
    allocation code generation objects inheriting from BaseCode.
    """
    functionality = []
    functionality_by_name = defaultdict(list)

    @classmethod
    def register(cls, funct):
        if funct not in cls.functionality:
            cls.functionality.append(funct)
            cls.functionality_by_name[funct.name].append(funct)

    @classmethod
    def get_funct(cls, funct_name):
        return cls.functionality_by_name[funct_name]
