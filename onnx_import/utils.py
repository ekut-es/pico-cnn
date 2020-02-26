from typing import Any, Text, Iterable, List, Dict, Sequence, Optional, Tuple, Union
from functools import reduce

__author__ = "Christoph Gerum (University of Tuebingen, Chair for Embedded Systems)"

def reduce_mult(data : Iterable[int]) -> int:
    return reduce(lambda x, y: x * y, data, 1)
