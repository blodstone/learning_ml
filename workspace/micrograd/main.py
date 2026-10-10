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
        self.grad = 1
        sorted_nodes = []

        def dfs(v):
            self._visited.add(v)
            for child in v.children:
                if child not in self._visited:
                    dfs(child)
            sorted_nodes.append(v)

        dfs(self)
        for node in sorted_nodes[::-1]:
            node._backward()

    def __add__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.value + other.value)
        out.children = [self, other]

        def backward():
            self.grad += out.grad * 1  # the derivative of x + y with respect to x is 1
            other.grad += out.grad * 1  # the derivative of x + y with respect to y is 1

        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if type(other) == Value else Value(other)
        out = Value(self.value * other.value)
        out.children = [self, other]

        def backward():
            self.grad += (
                out.grad * other.value
            )  # the derivative of x * y with respect to x is y
            other.grad += out.grad * self.value

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def sin(self):
        out = Value(math.sin(self.value))
        out.children = [self]

        def backward():
            self.grad += out.grad * math.cos(self.value)

        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.value))
        out.children = [self]

        def backward():
            print("I am inside.")
            self.grad += out.grad * math.exp(self.value)
            print(out.grad)

        out._backward = backward
        return out

    def __pow__(self, const):
        out = Value(self.value**const)
        out.children = [self]

        def backward():
            self.grad += out.grad * const * (self.value**const - 1)

        out._backward = backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return self**-1 * other


# %%
# Primal values
v_1 = x = Value(value=1.5)
v0 = y = Value(value=0.5)
v1 = v_1 / v0
v2 = v1.sin()
v3 = v0.exp()
v4 = v1 - v3
v5 = v2 + v4
v6 = v5 * v4
v6.backward()
# %%
print(v6)
print(v5)
print(v4)
print(v3)
print(v2)
print(v1)
print(v0)
print(v_1)
# %%
v1 = x / y
v2 = math.sin(v1)
v5 = v2 + v4
v6 = v5 * v4
loss = v6
# %%
# Adjoint values
# Simulate run backward on v6
a = Value(5)
b = a.exp()
b.backward()
print(b)
print(a)
