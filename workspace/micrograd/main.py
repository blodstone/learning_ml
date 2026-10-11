# %%
import math
from typing import Optional

import numpy as np


# %%
class Value:
    def __init__(self, value: float):
        self.value = value
        self.grad = 0
        self._backward = lambda: None

    def __div__(self, other: Value | float):
        if type(other) != Value:
            other = Value(other)

        def backward():
            pass


# %%
# Primal values
x = Value(value=2)
y = Value(value=3)
v1 = x / y
v2 = math.sin(v1)
v3 = math.exp(y)
v4 = v1 - v3
v5 = v2 + v4
v6 = v5 * v4
loss = v6
# %%
# Adjoint values
# Simulate run backward on v6


grad_v6 = 1
grad_v5 = grad_v6 * v5
