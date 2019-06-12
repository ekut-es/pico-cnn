from collections import defaultdict

class OperationRegistry(object):
    ops  = []
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
