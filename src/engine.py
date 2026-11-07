import math

class Value:
    def __init__(self,data, _children=(), _op='', label = ''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __neg__(self):
        return self * (-1)
    
    def __add__(self,other):
        self = self if isinstance(self,Value) else Value(self)
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data + other.data), (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self,other):
        return self + (-other)

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value((self.data * other.data), (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        if other.data == 0.0:
            raise ZeroDivisionError("Division by zero is invalid!")
        
        out = Value((self.data / other.data),(self,other), "/")
        def _backward():
            self.grad += (1/other.data) * out.grad
            other.grad += -(self.data/(other.data ** 2)) * out.grad
        out._backward = _backward 
        return out
    
    def exp(self):
        out = Value((math.exp(self.data)), (self,), "exp")
        def _backward():
            self.grad += out.grad * out.data
        return out
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) /( math.exp(2* n)+1)
        out = Value(t, (self,) , "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int,float))
        out = Value(self.data ** other, (self,), "f'**{other}'")

        def _backward():
            self.grad += out.grad * (self.data ** (other-1) ) * other
        
        out._backward = _backward
        return out

    def __rmul__(self,other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self,other):
        return other + (-self)

    def __rtruediv__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        return (other/self)

    def __rpow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(other ** self.data, (self,), f'{other}**')
        def _backward():
            if other != 0:
                self.grad += out.grad * out.data * math.log(other)
        out._backward = _backward
        return out

    def __rmod__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data % self.data, (self, other), '%')
        def _backward():
            other.grad += out.grad
            self.grad += -(other.data // self.data) * out.grad
        out._backward = _backward
        return out

    def __rfloordiv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.data // self.data, (self, other), '//')

    def __rdivmod__(self, other):
        return (self.__rfloordiv__(other), self.__rmod__(other))

    # The operators below have no useful derivative, so they do not propagate
    # gradients. Bitwise ones need int data; @ raises TypeError for plain scalars.
    def __rmatmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.data @ self.data, (self, other), '@')

    def __rlshift__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.data << self.data, (self, other), '<<')

    def __rrshift__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.data >> self.data, (self, other), '>>')

    def __rand__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.data & self.data, (self, other), '&')

    def __ror__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.data | self.data, (self, other), '|')

    def __rxor__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(other.data ^ self.data, (self, other), '^')

    def backward(self):
        topo = []
        def build_topo(node):
            if node not in topo:
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()