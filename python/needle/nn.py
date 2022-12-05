"""The module.
"""
from typing import List
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
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features), device=device, dtype=dtype)
        self.bias = Parameter(init.kaiming_uniform(out_features, 1).reshape(
            (1, out_features)), device=device, dtype=dtype) if bias else None

    def forward(self, X: Tensor) -> Tensor:

        y = X @ self.weight
        if self.bias:
            y += self.bias.broadcast_to(y.shape)
        return y


class Flatten(Module):
    def forward(self, X):

        batch_size = X.shape[0]
        new_dim = 1
        for i in X.shape[1:]:
            new_dim *= i
        return X.reshape((batch_size, new_dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:

        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:

        return ops.tanh(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:

        e = ops.exp(-x)
        return e / (e + e**2)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:

        for mod in self.modules:
            x = mod(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):

        batch_size = y.shape[0]
        y_one_hot = init.one_hot(
            logits.shape[1], y, device=logits.device, dtype=logits.dtype)
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

        self.weight = Parameter(init.ones(self.dim),
                                device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            batch_size = x.shape[0]
            mean = x.sum(axes=0) / batch_size
            x_mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ((x - x_mean) ** 2).sum(axes=0) / batch_size
            x_var = var.reshape((1, self.dim)).broadcast_to(x.shape)
            self.running_mean = ((1 - self.momentum) *
                                 self.running_mean + self.momentum * mean).data
            self.running_var = ((1 - self.momentum) *
                                self.running_var + self.momentum * var).data
        else:
            x_mean = self.running_mean.reshape(
                (1, self.dim)).broadcast_to(x.shape)
            x_var = self.running_var.reshape(
                (1, self.dim)).broadcast_to(x.shape)
        norm = (x - x_mean) / ((x_var + self.eps) ** 0.5)
        return self.weight.reshape((1, self.dim)).broadcast_to(x.shape) * norm + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose(
            (2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(self.dim),
                                device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(self.dim), device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.shape[0]
        x_mean = (x.sum(axes=1) / self.dim).reshape((batch_size, 1)
                                                    ).broadcast_to(x.shape)
        x_var = ((x - x_mean) ** 2).sum(axes=1).reshape((batch_size, 1)
                                                        ).broadcast_to(x.shape) / self.dim
        norm = (x - x_mean) / ((x_var + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            # prob = Tensor(np.random.binomial(n=1, p=1-self.p, size=x.shape))
            prob = init.randb(*x.shape, p=1-self.p,
                              device=x.device, dtype=x.dtype)
            return x * prob / (1-self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:

        return self.fn(x) + x


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = kernel_size * kernel_size * out_channels
        self.weight = Parameter(init.kaiming_uniform(
            fan_in, fan_out, shape=weight_shape, device=device, dtype=dtype))
        bias_bound = 1.0 / (in_channels * kernel_size**2)**0.5
        self.bias = Parameter(init.rand(out_channels, low=-bias_bound,
                              high=bias_bound, device=device, dtype=dtype)) if bias else None

    def forward(self, x: Tensor) -> Tensor:

        # NCHW -> NHWC
        x = x.transpose((1, 2)).transpose((2, 3))
        N, H, W, C = x.shape
        padding = self.kernel_size // 2
        activation = ops.conv(x, self.weight, self.stride, padding)
        if self.bias:
            b = self.bias.reshape((1, 1, 1, self.out_channels))
            b = b.broadcast_to(activation.shape)
            activation += b
        activation = activation.transpose((1, 3)).transpose((2, 3))
        return activation


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.nonlinearity = nonlinearity
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        bound = (1 / hidden_size)**0.5
        self.W_ih = Parameter(init.rand(
            input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(
            hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(
            hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(
            hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """

        batch_size = X.shape[0]
        if h is None:
            h = init.zeros(batch_size, self.hidden_size,
                           device=self.device, dtype=self.dtype)
        h_next = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih:
            bias_ih = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(
                (batch_size, self.hidden_size))
            bias_hh = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(
                (batch_size, self.hidden_size))
            h_next = h_next + bias_ih + bias_hh
        if self.nonlinearity == 'tanh':
            h_next = ops.tanh(h_next)
        elif self.nonlinearity == 'relu':
            h_next = ops.relu(h_next)
        return h_next


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.rnn_cells = [
            RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for i in range(1, num_layers):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """

        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size,
                            device=self.device, dtype=self.dtype)
        X_t = ops.split(X, axis=0)
        h_last_time = list(ops.split(h0, axis=0))
        h_last_layer = []
        for t in range(seq_len):
            first_layer_input = X_t[t]
            last_layer_h = 0
            for l in range(self.num_layers):
                rnn_cell = self.rnn_cells[l]
                if l == 0:
                    last_layer_h = rnn_cell(first_layer_input, h_last_time[l])
                else:
                    last_layer_h = rnn_cell(last_layer_h, h_last_time[l])
                h_last_time[l] = last_layer_h
            h_last_layer.append(last_layer_h)
        return ops.stack(h_last_layer, axis=0), ops.stack(h_last_time, axis=0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        bound = (1 / hidden_size)**0.5
        self.W_ih = Parameter(init.rand(
            input_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(
            hidden_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(
            4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(
            4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """

        batch_size = X.shape[0]
        if h is None:
            h = (init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype),
                 init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype))
        h0, c0 = h
        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih:
            bias_ih = self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to(
                (batch_size, 4*self.hidden_size))
            bias_hh = self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to(
                (batch_size, 4*self.hidden_size))
            gates = gates + bias_ih + bias_hh
        gates = gates.reshape((batch_size, 4, self.hidden_size))
        i, f, g, o = ops.split(gates, axis=1)
        input_gate = Sigmoid()(i)
        forget_gate = Sigmoid()(f)
        g_gate = Tanh()(g)
        output_gate = Sigmoid()(o)
        c_next = forget_gate * c0 + input_gate * g_gate
        h_next = output_gate * Tanh()(c_next)
        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.lstm_cells = [
            LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for i in range(1, num_layers):
            self.lstm_cells.append(
                LSTMCell(hidden_size, hidden_size, bias, device, dtype))

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """

        seq_len, bs, input_size = X.shape
        if h is None:
            h = (init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype),
                 init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype))
        h0, c0 = h
        X_t = ops.split(X, axis=0)
        h_last_time = list(ops.split(h0, axis=0))
        c_last_time = list(ops.split(c0, axis=0))
        h_last_layer = []
        for t in range(seq_len):
            first_layer_input = X_t[t]
            last_layer_h = 0
            for l in range(self.num_layers):
                lstm_cells = self.lstm_cells[l]
                if l == 0:
                    last_layer_h, last_layer_c = lstm_cells(
                        first_layer_input, (h_last_time[l], c_last_time[l]))
                else:
                    last_layer_h, last_layer_c = lstm_cells(
                        last_layer_h, (h_last_time[l], c_last_time[l]))
                h_last_time[l] = last_layer_h
                c_last_time[l] = last_layer_c
            h_last_layer.append(last_layer_h)
        return ops.stack(h_last_layer, axis=0), (ops.stack(h_last_time, axis=0), ops.stack(c_last_time, axis=0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(
            num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """

        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x.reshape(
            (seq_len * bs,)), device=self.device, dtype=self.dtype)
        output = one_hot @ self.weight
        return output.reshape((seq_len, bs, self.embedding_dim))
