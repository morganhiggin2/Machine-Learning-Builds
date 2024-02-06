import torch
import torch.nn
import pandas
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plot

class LinearRegression(torch.nn.Module):
    def __init__(self, lr):
        super().__init__()
        
        self.net = torch.nn.LazyLinear(1)

    def forward(self, X):
        return self.net(X)

#data = pandas.read_csv('../data/simple_1.csv', index_col=False)
#generate custom data 
#x, y = sklearn.datasets.make_regression(n_samples=20, n_features=1, bias=0, noise=0, random_state=2)
x = np.arange(0, 20) + np.random.uniform(0, 1, 20) * 10
y = 5.5 * x + 0.2

#data_tensor = torch.tensor(np.array([x[:,0], y]), requires_grad=True, dtype=torch.float32)

data_tensor = torch.tensor(np.array([x, y]), requires_grad=True, dtype=torch.float32)
data_tensor = data_tensor.reshape(data_tensor.size(1), data_tensor.size(0))


model = LinearRegression(lr=0.01)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-2
)
loss_fn = torch.nn.L1Loss()


for epoch in range(50):
    for i in range(data_tensor.size(0)):
        optim.zero_grad()

        output = model(data_tensor[i, 0].unsqueeze(0))
  
        loss = loss_fn(output, data_tensor[i, 1].unsqueeze(0))
        loss.backward()
   
        optim.step()

data = data_tensor.detach().numpy()
predicted_values = data_tensor[:,0].detach().apply_(lambda x: model(torch.tensor(x).unsqueeze(0))).numpy()
#predicted_values = np.array([model(x.unsqueeze(0)).detach() for x in data_tensor[:, 0]])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c='black')
axs[1].scatter(data[:,0], predicted_values, c='orange')
plot.show()
