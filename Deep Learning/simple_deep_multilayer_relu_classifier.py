import torch
import torch.nn
import pandas
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plot
from matplotlib import colors

class LinearClassifier(torch.nn.Module):
    def __init__(self, num_hidden, num_outputs):
        super().__init__()

        hidden_net = torch.nn.LazyLinear(num_hidden)
        relu = torch.nn.ReLU()
        linear_net = torch.nn.LazyLinear(num_outputs)
        flatten = torch.nn.Flatten()

        self.net = torch.nn.Sequential(flatten, hidden_net, relu, linear_net)

    def forward(self, X):
        return self.net(X)

n = 2
h = n + 1
n_c = n

X, Y = sklearn.datasets.make_classification(n_samples=50, n_features=n, n_redundant=0, n_informative=n, n_repeated=0, n_classes=n_c, random_state=4)


data_tensor = torch.zeros((X.shape[0], X.shape[1] + 1), dtype=torch.float32)
data_tensor[:,0:n] = torch.from_numpy(X).to(data_tensor)
data_tensor[:,n] = torch.from_numpy(Y).to(data_tensor).unsqueeze(0)

labels = ['x1', 'x2', 'x3', 'c']

model = LinearClassifier(num_hidden=h, num_outputs=n)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-2
)
loss_fn = torch.nn.CrossEntropyLoss()

data_tensor.requires_grad_(True)

for epoch in range(500):
    output = model(data_tensor[:,0:n])

    loss = loss_fn(output, data_tensor[:,n].long())

    optim.zero_grad()
   
    loss.backward()

    optim.step()

    if epoch % 10 == 0:
        print(str(epoch) + " " + str(loss))

data = data_tensor.detach().numpy()

predicted_classes = model(data_tensor[:,0:n]).argmax(1)

print(predicted_classes.numpy())
print(data[:,n])
print(predicted_classes == data_tensor[:,n])

color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])#color_map[data[:,n]])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())#color_map[predicted_classes])
plot.show()
