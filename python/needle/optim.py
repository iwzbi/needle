"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        for p in self.params: self.u[p] = 0

    def step(self):
        for p in self.params:
            regula = p.grad + p * self.weight_decay
            self.u[p] = (self.momentum * self.u[p] + (1 - self.momentum) * regula).data
            p.cached_data -= self.lr * self.u[p].cached_data


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

        for p in self.params: 
            self.m[p] = 0
            self.v[p] = 0

    def step(self):
        self.t += 1
        for p in self.params:
            regula = p.grad + p * self.weight_decay
            m = self.beta1 * self.m[p] + (1 - self.beta1) * regula
            v = self.beta2 * self.v[p] + (1 - self.beta2) * (regula ** 2)
            self.m[p] = m.data
            self.v[p] = v.data
            m = (m / (1 - self.beta1 ** self.t)).data
            v = (v / (1 - self.beta2 ** self.t)).data
            p.cached_data -= self.lr * (m / (v ** 0.5 + self.eps)).cached_data
