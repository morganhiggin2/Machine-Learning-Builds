import torch
import torch.nn
import pandas
import numpy
import random
import sklearn.datasets
import matplotlib.pyplot as plot
from matplotlib import colors

class SequencePredictor(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.lin = torch.nn.Linear(in_features=num_features, out_features=1)

        self.net = torch.nn.Sequential(self.lin)

    def forward(self, X):
        return self.net(X)

n = 3
d = 2
h = n + 1
e_loss = 1e-4
use_gpu = True
file_name = "data/simple_seq_target_1_data.csv"
num_data = 100

#generate sequencial 2d data based on a function
data = numpy.array([[0, 0], [-4, 3], [5, 4]])
targets = numpy.array([])

for i in range(num_data):
    data = numpy.concatenate((data, numpy.array([[random.randint(-10, 10), random.randint(-10, 10)]])), axis=0)

for i in range(num_data - n):
    x = data[i,0] + 2 * data[i+1,0] - data[i+2,0]
    y = -3 * data[i,1] + data[i+1,1] + 2 * data[i+2,1] 
    targets = numpy.concatenate((targets, numpy.array([x + y])), axis=0)

#reshape data to be a linear combination of sequencial data

sequential_data = numpy.empty((num_data-3, n*d))

for i in range(num_data - n):
    sequential_data[i,:] = numpy.ndarray.flatten(data[i:i+n,:])

data_tensor = torch.tensor(sequential_data, dtype=torch.float32)
target_tensor = torch.tensor(targets, dtype=torch.float32)

target_tensor = torch.unsqueeze(target_tensor, dim=1)

model = SequencePredictor(num_features=n*d)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-2
)
loss_fn = torch.nn.MSELoss()

data_tensor.requires_grad_(True)
target_tensor.requires_grad_(True)

if use_gpu:
    if torch.cuda.device_count() >= 1:
        model.net = model.net.to(torch.device(f'cuda:{0}'))
        data_tensor = data_tensor.to(torch.device(f'cuda:{0}'))
        target_tensor = target_tensor.to(torch.device(f'cuda:{0}'))

prev_loss = -1
curr_loss = -1

for epoch in range(500):
    output = model(data_tensor)
    
    loss = loss_fn(output, target_tensor)
   
    optim.zero_grad()
   
    loss.backward()

    optim.step()

    if epoch % 10 == 0:
        print(str(epoch) + " " + str(loss))

    curr_loss = loss.detach().item()

    if abs(curr_loss - prev_loss) < e_loss:
        break

    prev_loss = curr_loss

if use_gpu and torch.cuda.device_count() >= 1:
    data = data_tensor.cpu().detach().numpy()
    predicted_targets = model(data_tensor).cpu()
else:
    data = data_tensor.detach().numpy()
    predicted_targets = model(data_tensor)

predicted_targets = model(data_tensor)

print(torch.round((target_tensor - predicted_targets), decimals=1))
'''
color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])#color_map[data[:,n]])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())#color_map[predicted_classes])
plot.show()'''
