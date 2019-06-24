from collections import defaultdict


class OperationRegistry(object):
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
