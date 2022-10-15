"""The module.
"""
from operator import mul
from typing import List, Callable, Any
from xml.sax.handler import feature_external_ges
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype)
        self.bias = Parameter(init.kaiming_uniform(out_features, 1).reshape((1, out_features)), device=device, dtype=dtype) if bias else None

    def forward(self, X: Tensor) -> Tensor:
        y =  X @ self.weight
        if self.bias: y += self.bias.broadcast_to(y.shape)
        return y



class Flatten(Module):
    def forward(self, X):
        batch_size = X.shape[0]
        new_dim = 1
        for i in X.shape[1:]: new_dim *= i
        return X.reshape((batch_size, new_dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.modules: x = mod(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        batch_size = y.shape[0]
        y_one_hot = init.one_hot(logits.shape[1], y)
        expzsum = ops.log(ops.exp(logits).sum(axes=1))
        zy = ops.summation(logits * y_one_hot, axes=1)
        loss = ops.summation(expzsum - zy)
        return loss / batch_size



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(self.dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)


    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_size = x.shape[0]
            mean = x.sum(axes=0) / batch_size
            x_mean = mean.broadcast_to(x.shape)
            var = ((x - x_mean) ** 2).sum(axes=0) / batch_size
            x_var = var.broadcast_to(x.shape)
            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean).data
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var).data
        else:
            x_mean = self.running_mean.broadcast_to(x.shape)
            x_var = self.running_var.broadcast_to(x.shape)
        norm = (x - x_mean) / ((x_var + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)   
            


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(self.dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x_mean = (x.sum(axes=1) / self.dim).reshape((batch_size, 1)).broadcast_to(x.shape)
        x_var = ((x - x_mean) ** 2).sum(axes=1).reshape((batch_size, 1)).broadcast_to(x.shape) / self.dim
        norm = (x - x_mean) / ((x_var + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            #prob = Tensor(np.random.binomial(n=1, p=1-self.p, size=x.shape))
            prob = init.randb(*x.shape, p=1-self.p)
            return x * prob / (1-self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x



