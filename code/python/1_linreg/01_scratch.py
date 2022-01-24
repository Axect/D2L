import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise"""
    n = Normal(0, 1)
    X = n.sample((num_examples, len(w)))
    y = X @ w + b
    
    n_eps = Normal(0, 0.01)
    eps = n_eps.sample((num_examples,))
    y += eps
    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
features, labels = synthetic_data(true_w, true_b, 1000)

plt.figure(figsize=(10, 6), dpi=120)
plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy());
plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

di = data_iter(batch_size, features, labels)
print(type(di))

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    """The Linear regression model"""
    return X @ w + b

def squared_loss(y_hat, y):
    """Squared Loss"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X, w, b)
        l = loss(y_hat, y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        pred = net(features, w, b)
        train_l = loss(pred, labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')