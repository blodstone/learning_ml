# %%
from __future__ import annotations

import math


# %%
class Value:
    def __init__(self, value: float):
        self.value = value
        self.grad = 0
        self._backward = lambda: None
        self.children = []
        self._visited = set()

    def __repr__(self):
        return f"Value(value={self.value}, grad={self.grad})"

    def backward(self):
        sorted_nodes = []

        def dfs(v):
            self._visited.add(v)
            for child in v.children:
                if child not in self._visited:
                    dfs(child)
            sorted_nodes.append(v)

        for child in self.children:
            if child not in self._visited:
                dfs(child)
        for node in sorted_nodes[::-1]:
            node._backward()

    def __add__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.value + other.value)
        out.children = [self, other]

        def backward():
            self.grad += out.grad * 1  # the derivative of x + y with respect to x is 1
            other.grad += out.grad * 1  # the derivative of x + y with respect to y is 1

        self._backward = backward
        return out

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.value * other.value)
        out.children = [self, other]

        def backward():
            self.grad += (
                out.grad * other.value
            )  # the derivative of x * y with respect to x is y
            other.grad += out.grad * self.value

        self._backward = backward
        return out


# %%
# Primal values
x = Value(value=2)
y = Value(value=3)
z = x + y
print(z.value)
z.grad = 1
z.backward()
print(x)
# %%
print(x.grad)
# %%
v1 = x / y
v2 = math.sin(v1)
v5 = v2 + v4
v6 = v5 * v4
loss = v6
# %%
# Adjoint values
# Simulate run backward on v6


grad_v6 = 1
grad_v5 = grad_v6 * v5
