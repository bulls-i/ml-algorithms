# Stochastic Gradient Descent
# Linear regression

import torch

import numpy as np 
w = torch.tensor(-10.0, requires_grad=True)
X = torch.arange(-3,3,0.1).view(-1, 1)
# Y = -3X + (gaussian noise)
Y = -3 * X + 0.3 * torch.randn(X.size())

# Y = 4 sinX + (gaussian noise)
# Y = 4 * torch.sin(X) + 0.4 * torch.randn(X.size())

import matplotlib.pyplot as plt 

plt.plot(X.numpy(), Y.numpy(), 'ro')
plt.show()

def criterion(y, y_):
    return torch.mean((y - y_) ** 2)

def forward(x):
    return w * x

# Create model
import torch.nn as nn
model = nn.Linear(1, 1)
model.weight.data = torch.tensor([[10.0]], requires_grad=True)
model.bias.data = torch.tensor(2.0, requires_grad=True)

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr = 0.01)
lr = 0.1

# Train
for i in range(4):
    for (x,y) in zip(X, Y):
        y_ = model(X)
        loss = criterion(Y, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(loss.item())
    p = model(X)
    plt.plot(X.numpy(), p.data.numpy())
    plt.plot(X.numpy(), Y.numpy(), 'ro')
    plt.show()


