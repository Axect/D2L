# 3. Linear Neural Network

[TOC]

## 3.1 Linear Regression

### 3.1.1 Basic Elements of Linear Regression

#### 3.1.1.1 Linear Model

**Basic components**

* Input: $\mathbf{x} \in \mathbb{R}^d$

* Target: $y \in \mathbb{R}$

* Data: $(\mathbf{x}^{(i)}, y^{(i)}) \in \mathbb{R}^d \times \mathbb{R}$

* Weight: $\mathbf{w} \in \mathbb{R}^d$

* Bias: $b \in \mathbb{R}$

  

**Single prediction**
$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$



**Collection of predictions**

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b
$$

* Design matrix: $\mathbf{X} \in \mathbb{R}^{n \times d}$
* Target vector: $\mathbf{y} \in \mathbb{R}^n$

#### 3.1.1.2 Loss Function

**Squared Error**
$$
l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)} \right)^2
$$



**Mean Squared Error (MSE)**
$$
L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n l^{(i)}(\mathbf{w}, b)
$$



**Best-fit parameters**
$$
\mathbf{w}^*,b^* = \underset{\mathbf{w},b}{\text{argmin}}\, L(\mathbf{w}, b)
$$

#### 3.1.1.3 Analytic Solution

We can find the below relation is true.
$$
\underset{\mathbf{w},b}{\text{argmin}}\, L(\mathbf{w}, b) = \underset{\mathbf{w},b}{\text{argmin}}\, \Vert \mathbf{y} - \mathbf{X} \mathbf{w} - b \Vert^2
$$

Let re-define $\mathbf{X}$ and $\mathbf{w}$ as below.

```python
# Pytorch
X = torch.column_stack([torch.ones_like(X[:,0]), X])
w = torch.cat([b.unsqueeze(0), w])
```

Then we can rewrite the above equation as below.
$$
\underset{\mathbf{w}}{\text{argmin}}\, L(\mathbf{w}) = \underset{\mathbf{w}}{\text{argmin}}\, \Vert \mathbf{y} - \mathbf{X} \mathbf{w}\Vert^2
$$

With an assumption that $\mathbf{X}$ has a full column rank[^1], then we can find the analytic solution:
$$
\mathbf{w}^* = \left(\mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{y}
$$

-----

[^1]: If you need more precise proof, then see [Axect's Slides - ESL Chapter 3 Linear Regression - OLS Estimation](https://axect.github.io/Slides/ML/ESL/chap3/03_linreg_1.html#34)

#### 3.1.1.4 Minibatch Stochastic Gradient Descent

**Motivation**

* Although we can't solve the models analytically, we can still train models via *Gradient-Descent*.
* But original Gradient-Descent needed computation for whole dataset for every update. In practice, this can be extremely slow.
* To solve it, we will often settle for sampling a random minibatch of examples every time we need to compute the update, a variant called *minibatch stochastic gradient descent*.



**Algorithm**

1. Randomly sample a minibatch $\mathcal{B}$ consisting of a fixed number of training examples.
2. Compute the derivative of the average loss on the minibatch with regard to the model parameters.
3. Multiply the gradient by a predetermined positive value $\eta$ and subtract the resulting term from the current parameter values.



We describe this algorithm as below equation:
$$
\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i\in\mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w})
$$
There are two *hyperparameters*[^2] in this equation:

* $\eta$ : Learning rate
* $|\mathcal{B}|$ : Batch size

We adjust hyperparameter tuning based on the results of the training loop as assessed on a separate validation dataset.

-----

[^2]: Parameters which are tunable but not updated in the training loop.

### 3.1.2 Vectorization for Speed

* In python, iteration speed is too slow. So, we should use vectorized code instead.
* Here is the benchmark[^3]

![Scientific Bench](large_plot.png)

[^3]: For more details, refer to [Scientific Bench by Axect](https://github.com/Axect/Scientific_Bench/tree/master/Basic/sum)

```rust
// Rust for loop
fn for_sum(v: &[f64]) -> f64 {
    let mut s = 0f64;
    for t in v {
        s += *t;
    }
    s
}
```

```rust
// Rust Chunk sum
fn chunk_sum(v: &[f64]) -> f64 {
    v.chunks(8).map(|t| t.iter().sum::<f64>()).sum()
}
```

```rust
// Rust SIMD (Old)
use packed_simd2::f64x8;

fn simd_sum(v: &[f64]) -> f64 {
    let mut sum = f64x8::splat(0.);
    for i in (0 .. x.len()).step_by(8) {
        sum += f64x8::from_slice_unaligned(&v[i..]);
    }
    sum.sum()
}
```

```rust
// Rust SIMD (Nightly)
#![feature(portable_simd)]
use std::simd::f64x8;

fn simd_sum(v: &[f64]) -> f64 {
    let mut sum = f64x8::splat(0.);
    for i in (0 .. x.len()).step_by(8) {
        sum += f64x8::from_slice(&v[i..]);
    }
    sum.as_array().iter().sum()
}
```

```julia
# Julia SIMD
function sum_simd(x::S) where { T <: Number, S <: AbstractVector{T} }
    s = zero(eltype(x));
    @inbounds @simd for t in x
        s += t
    end
    return s
end
```

### 3.1.3 The Normal Distribution and Squared Loss

* Let $\varepsilon \sim \mathcal{N}(0,\sigma^2)$ then
  $$
  P(\mathbf{y} | \mathbf{X}) = \prod_{i=1}^n p(y^{(i)} | \mathbf{x}^{(i)}) = \exp \left[ -\frac{1}{2\sigma^2}\sum_{i=1}^n (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2 \right]
  $$

* Thus, find $\mathbf{w}$ to maximize this likelihood is same work with minimize $\sum (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2$
  $$
  \mathbf{w}^* = \underset{\mathbf{w}}{\text{argmin}}\left\{\lVert\mathbf{y} - \mathbf{X}\mathbf{w} \rVert^2\right\} = \underset{\mathbf{w}}{\text{argmax}}\,\left\{P(\mathbf{y}|\mathbf{X})\right\}
  $$

* OLS with an assumption of normal error = MLE

-----

## 3.2 Linear Regression Implementation from Scratch

**1) Generate Data**

```python
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
```

**2) Reading the Dataset**

```python
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
```

* `yield` : Generator syntax - return iterator

**3) Initializing Model Parameters**

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

**4) Defining the Model**

```python
def linreg(X, w, b):
    """The Linear regression model"""
    return X @ w + b
```

**5) Defining the Loss Function**

```python
def squared_loss(y_hat, y):
    """Squared Loss"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

**6) Defining the Optimization Algorithm**

```python
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

**7) Training**

```python
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
```

## 3.3 Concise Implementation of Linear Regression

### A. PyTorch

```python
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================================================================
# Define Dataset & DataLoader
# ==============================================================================
class SyntheticData(Dataset):
    def __init__(self, w, b, num_examples):
        n = Normal(0, 1)
        X = n.sample((num_examples, len(w)))
        y = X @ w + b

        n_eps = Normal(0, 0.01)
        eps = n_eps.sample((num_examples,))
        y += eps

        self.X = X
        self.y = y.reshape(-1, 1)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

true_w = torch.tensor([2, -3.4])
true_b = torch.tensor(4.2)
num_examples = 1000

train_data = SyntheticData(true_w, true_b, num_examples)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# ==============================================================================
# Define Neural Network
# ==============================================================================
class LinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1)
        )
    
    def forward(self, X):
        return self.net(X)

# ==============================================================================
# Define Loss & Optimizer
# ==============================================================================
net = LinReg()
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)

# ==============================================================================
# Training
# ==============================================================================
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in train_loader:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(train_data.X), train_data.y)
    print(f'epoch {epoch + 1}: loss {l.item():f}')
```

