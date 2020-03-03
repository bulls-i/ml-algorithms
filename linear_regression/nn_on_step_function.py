# Modeling step function using neural networks

import torch 
import torch.nn as nn
import torch.optim as optim

# Step function (-1 to 1 => 1)
def f(x):
    a = (-1 <= x)
    b = (x <= 1)
    res = torch.zeros(x.size(), dtype=int)
    res[a == b] = 1
    return res

# Neural net with hidden layer of size 20
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.l1 = nn.Linear(1,20)
        self.l2 = nn.Linear(20,1)
    
    def forward(self, x):
        ### ------------------
        x = torch.relu(self.l1(x))
        # x = torch.sigmoid(self.l1(x))
        # x = self.l1(x)
        ### ------------------
        # x = torch.relu(self.l2(x))
        x = torch.sigmoid(self.l2(x))
        # x = self.l2(x)
        ### ------------------

        return x

def criterion(y, y_):
    return torch.mean((y - y_) ** 2)

# Input data
X = torch.arange(-5,5,0.1).view(-1,1)
y = f(X)

model = NNet()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

# Training
for i in range(50001):
    y_ = model(X)
    loss = criterion(y, y_)
    # print(y, y_)
    # print('Loss', loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i % 5000 == 0:
        p = model(X)
        print('Iteration', i, 'Loss', loss.item())
        import matplotlib.pyplot as plt 
        plt.plot(X.numpy(), y.numpy())
        plt.plot(X.numpy(), p.data.numpy(), 'ro')
        plt.show()
    
print('Done')