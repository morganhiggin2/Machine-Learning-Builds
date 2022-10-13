import torch
import torch.nn
import pandas
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plot
from matplotlib import colors

class LinearClassifier(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        linear_net = torch.nn.Linear(num_inputs, num_outputs)
        self.flatten = torch.nn.Flatten()
        
        self.net = torch.nn.Sequential(linear_net)
        torch.nn.init.xavier_uniform(linear_net.weight)

    def forward(self, X):
        logits = self.net(X)
        return logits

n = 2
n_c = n
lr = 1
b = 0.5
e_loss = 0.01

X, Y = sklearn.datasets.make_classification(n_samples=50, n_features=n, n_redundant=0, n_informative=n, n_repeated=0, n_classes=n_c, random_state=4)

data_tensor = torch.zeros((X.shape[0], X.shape[1] + 1), dtype=torch.float32)
data_tensor[:,0:n] = torch.from_numpy(X).to(data_tensor)
data_tensor[:,n] = torch.from_numpy(Y).to(data_tensor).unsqueeze(0)

n_inputs = data_tensor.size(1) - 1

labels = ['x1', 'x2', 'x3', 'c']

model = LinearClassifier(num_inputs=n_inputs, num_outputs=n)
optim = torch.optim.LBFGS(
    [{'params': model.parameters()}],
    lr=lr
)

loss_fn = torch.nn.CrossEntropyLoss()
prev_loss = -1
curr_loss = -1

def weight_loss_fn(model):
    reg = 0

    for param in model.parameters():
        reg += b * (param ** 2).sum()
    
    return reg

def closure():
    optim.zero_grad()

    output = model(data_tensor[:,0:n])
    loss = loss_fn(output, data_tensor[:,n].long()) + weight_loss_fn(model)

    loss.backward()

    curr_loss = loss.detach().item()

    print(str(epoch) + " " + str(loss))
   
    return loss

data_tensor.requires_grad_(True)

for epoch in range(500):
    optim.step(closure)

    if abs(curr_loss - prev_loss) < e_loss:
        break

    prev_loss = curr_loss

data = data_tensor.detach().numpy()

predicted_classes = model(data_tensor[:,0:n]).argmax(1)

print(predicted_classes.numpy())
print(data[:,n])
print(predicted_classes == data_tensor[:,n])

color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())
plot.show()
