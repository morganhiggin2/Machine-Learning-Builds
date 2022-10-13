import torch
import torch.nn
import pandas
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plot
from matplotlib import colors

#https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be

class LinearClassifier(torch.nn.Module):
    def __init__(self, num_outputs):
        super().__init__()

        linear_net = torch.nn.LazyLinear(num_outputs)
        self.flatten = torch.nn.Flatten()

        self.net = torch.nn.Sequential(linear_net)

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits

n = 2
n_c = n

X, Y = sklearn.datasets.make_classification(n_samples=50, n_features=n, n_redundant=0, n_informative=n, n_repeated=0, n_classes=n_c, random_state=4)


data_tensor = torch.zeros((X.shape[0], X.shape[1] + 1), dtype=torch.float32)
data_tensor[:,0:n] = torch.from_numpy(X).to(data_tensor)
data_tensor[:,n] = torch.from_numpy(Y).to(data_tensor).unsqueeze(0)

labels = ['x1', 'x2', 'x3', 'c']

model = LinearClassifier(num_outputs=n)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-2
)
loss_fn = torch.nn.CrossEntropyLoss()

data_tensor.requires_grad_(True)

for epoch in range(5000):
    output = model(data_tensor[:,0:n])

    loss = loss_fn(output, data_tensor[:,n].long())

    optim.zero_grad()
   
    loss.backward()

    optim.step()

    if epoch % 10 == 0:
        print(str(epoch) + " " + str(loss))

data = data_tensor.detach().numpy()

#print(model(data_tensor[:,0:3]))
#print(data_tensor[:,3].unsqueeze(0).reshape(data_tensor.size(0), 1))
predicted_classes = model(data_tensor[:,0:n]).argmax(1)

print(predicted_classes.numpy())
print(data[:,n])
print(predicted_classes == data_tensor[:,n])

color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])#color_map[data[:,n]])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())#color_map[predicted_classes])
plot.show()
