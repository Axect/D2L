import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data
import torchvision
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH_DATASETS = os.environ.get("PATH_DATASETS")

# ==============================================================================
# Get Dataset & Define Transforms
# ==============================================================================
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root=PATH_DATASETS,
    train=True,
    transform=trans,
    download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root=PATH_DATASETS,
    train=False,
    transform=trans,
    download=True
)

print(mnist_train[0][0].shape)
print(mnist_test[0][1])

# ==============================================================================
# Convert integer label to text
# ==============================================================================
def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion_MNIST dataset"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

# ==============================================================================
# Define dataloader
# ==============================================================================
train_loader = DataLoader(
    mnist_train,
    batch_size=16,
    shuffle=True,
    num_workers=8
)
test_loader = DataLoader(
    mnist_test,
    batch_size=16,
    shuffle=False,
    num_workers=8
)

# ==============================================================================
# Visualize some images
# ==============================================================================
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.reshape(-1,)
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(train_loader))
show_images(X.reshape(-1, 28, 28), 2, 8, titles=get_fashion_mnist_labels(y))
plt.show()

# ==============================================================================
# Redefine batch size & Make iterator
# ==============================================================================
batch_size = 256
train_loader = DataLoader(
    mnist_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)
test_loader = DataLoader(
    mnist_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8
)

train_iter = iter(train_loader)
test_iter = iter(test_loader)

# ==============================================================================
# Initializing Model Parameters
# ==============================================================================
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# ==============================================================================
# Define softmax function
# ==============================================================================
def softmax(X):
    exp = torch.exp(X)
    return exp / exp.sum(dim=1, keepdim=True)

# Check
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(dim=1, keepdim=True))

# ==============================================================================
# Defining the Model
# ==============================================================================
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# ==============================================================================
# Defining the Loss function
# ==============================================================================
def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

# Check
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(cross_entropy(y_hat, y))

# ==============================================================================
# Classification Accuracy
# ==============================================================================
def accuracy(y_hat, y):
    """Compute the number of correct predictions"""
    return float((y_hat.argmax(dim=1) == y).float().sum())

# Check
accuracy(y_hat, y) / len(y)

def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset"""
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.shape[0])
    return metric[0] / metric[1]

class Accumulator:
    """For accumulating sums over `n` variables"""
    def __init__(self, n):
        self.data = torch.zeros(n)
    
    def add(self, *args):
        self.data = torch.tensor([a + float(b) for a, b in zip(self.data, args)])
    
    def reset(self):
        self.data.zero_()

    def __getitem__(self, idx):
        return self.data[idx]

evaluate_accuracy(net, test_iter)

# ==============================================================================
# Training
# ==============================================================================
def train_epoch_ch3(net, train_iter, loss, updater):
    """Train a model on the training set"""
    if isinstance(net, nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples    
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            l.backward()
            updater(X.shape[0])
        metric.add(l.item(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model (defined in Chapter 3)"""
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        pbar.set_description(
            f'Loss: {train_metrics[0]:.3f}, Acc: {test_acc.item():.3f}'
        )
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def updater(batch_size):
    return sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)