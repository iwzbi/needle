import numpy as np
import math
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('./python')
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

        def ConvBN(c_in, c_out, kernel_size, stride):
            return nn.Sequential(nn.Conv(c_in, c_out, kernel_size, stride, device=device, dtype=dtype),
                                 nn.BatchNorm2d(c_out, device=device, dtype=dtype), nn.ReLU())
        self.resnet9 = nn.Sequential(
            ConvBN(3, 16, 7, 4),
            ConvBN(16, 32, 3, 2),
            nn.Residual(nn.Sequential(
                ConvBN(32, 32, 3, 1), ConvBN(32, 32, 3, 1))),
            ConvBN(32, 64, 3, 2),
            ConvBN(64, 128, 3, 2),
            nn.Residual(nn.Sequential(
                ConvBN(128, 128, 3, 1), ConvBN(128, 128, 3, 1))),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 10, device=device, dtype=dtype)
        )

    def forward(self, x):

        return self.resnet9(x)


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()

        self.embed_layer = nn.Embedding(
            output_size, embedding_size, device, dtype)
        if seq_model == 'rnn':
            self.seq_layer = nn.RNN(
                embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_layer = nn.LSTM(
                embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear_layer = nn.Linear(
            hidden_size, output_size, device=device, dtype=dtype)

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """

        x_embedded = self.embed_layer(x)
        last_layer, last_time = self.seq_layer(x_embedded, h)
        seq_len, bs, hidden_size = last_layer.shape
        out = self.linear_layer(last_layer.reshape((seq_len*bs, hidden_size)))
        return out, last_time


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)
