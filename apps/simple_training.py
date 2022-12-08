import time
from models import *
from needle import backend_ndarray as nd
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('../python')

device = ndl.cpu()

### CIFAR-10 training ###


def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    if opt is None:
        model.eval()
    total_loss = 0
    total_example = len(dataloader.dataset)
    right_num = 0
    for x, y in dataloader:
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.detach().numpy()
        y_hat = np.argmax(logits.detach().numpy(), axis=1)
        right_num += np.sum(y_hat == y.numpy())
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    average_loss = total_loss / total_example
    accuray = right_num / total_example
    return accuray, average_loss


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr, weight_decay)
    for i in range(n_epochs):
        start_time = time.time()
        acc, loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt)
        end_time = time.time()
        print("train epoch{}: acc: {}, loss: {}, time cost{}".format(
            i, acc, loss, end_time-start_time))
    return acc, loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    acc, loss = epoch_general_cifar10(dataloader, model, loss_fn())
    print("evaluate: acc: {}, loss: {}".format(acc, loss))
    return acc, loss


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    if opt is None:
        model.eval()
    total_loss = 0
    right_num = 0
    nbatch, batch_size = data.shape
    last_time_h = None
    total_example = 0

    for i in range(0, nbatch-1, seq_len):
        x, target = ndl.data.get_batch(
            data, i, seq_len, device=device, dtype=dtype)
        real_len = target.shape[0]
        total_example += real_len
        out, last_time_h = model(x, last_time_h)
        if isinstance(last_time_h, tuple):
            last_time_h = (ndl.Tensor(last_time_h[0].detach().numpy(), device=device, dtype=dtype), ndl.Tensor(
                last_time_h[1].detach().numpy(), device=device, dtype=dtype))
        else:
            last_time_h = ndl.Tensor(
                last_time_h.detach().numpy(), device=device, dtype=dtype)
        loss = loss_fn(out, target)
        total_loss += loss.detach().numpy() * real_len
        y_hat = np.argmax(out.detach().numpy(), axis=1)
        right_num += np.sum(y_hat == target.numpy())
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    average_loss = total_loss / total_example
    accuray = right_num / total_example
    return accuray, average_loss


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
              device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr, weight_decay)
    for i in range(n_epochs):
        start_time = time.time()
        acc, loss = epoch_general_ptb(
            data, model, seq_len, loss_fn(), opt, clip, device, dtype)
        end_time = time.time()
        print("train epoch{}: acc: {}, loss: {}, time cost: {}".format(
            i, acc, loss, end_time-start_time))
    return acc, loss


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
                 device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    acc, loss = epoch_general_ptb(
        data, model, seq_len, loss_fn(), device=device, dtype=dtype)
    print("evaluate: acc: {}, loss: {}".format(acc, loss))
    return acc, loss


if __name__ == "__main__":
    # For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    # dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    # train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(
        corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary),
                          hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
