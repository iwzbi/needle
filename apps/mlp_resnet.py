import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    l1 = nn.Linear(in_features=dim, out_features=hidden_dim)
    n1 = norm(dim=hidden_dim)
    r1 = nn.ReLU()
    d1 = nn.Dropout(p=drop_prob)
    l2 = nn.Linear(in_features=hidden_dim, out_features=dim)
    n2 = norm(dim=dim)
    r2 = nn.ReLU()
    fn = nn.Sequential(l1, n1, r1, d1, l2, n2)
    residual = nn.Residual(fn)
    block = nn.Sequential(residual, r2)
    return block


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    l1 = nn.Linear(in_features=dim, out_features=hidden_dim)
    r1 = nn.ReLU()
    fn = [l1, r1]
    for i in range(num_blocks):
        fn.append(ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob))
    l2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)
    fn.append(l2)
    rn = nn.Sequential(*fn)
    return rn




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    if opt is None:
        model.eval()
    else:
        model.train() 
        opt.reset_grad()
    iteration = 0
    total_loss = 0
    total_example = len(dataloader.dataset)
    right_num = 0
    for x, y in dataloader:
        iteration += 1
        f = nn.Flatten()
        x = f(x)
        logits = model(x)
        loss_fn = nn.SoftmaxLoss()
        loss = loss_fn(logits, y)
        total_loss += loss
        y_hat = np.argmax(logits.numpy(), axis=1)
        right_num += np.sum(y_hat == y.numpy())
        if opt:
            loss.backward()
            opt.step()
    average_loss = total_loss.numpy() / iteration
    accuray = right_num / total_example
    return 1-accuray, average_loss



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    train_images = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    train_dataset = ndl.data.MNISTDataset(train_images, train_labels)
    test_dataset = ndl.data.MNISTDataset(test_images, test_labels)
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = MLPResNet(28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_acc, train_loss = epoch(train_dataloader, model, opt)
    test_acc, test_loss = epoch(test_dataloader, model)
    return train_acc, train_loss, test_acc, test_loss



if __name__ == "__main__":
    train_mnist(data_dir="../data")
