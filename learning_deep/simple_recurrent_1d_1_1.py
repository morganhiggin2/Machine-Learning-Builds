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

d = 1
e_loss = 1e-4
use_gpu = True
file_name = "data/simple_recurrent_pattern_1d.csv"
num_data = 100

#read pattern from csv file 
input_data_pandas = pandas.read_csv(file_name, delimiter=" ", header=None)
input_data = input_data_pandas.to_numpy().squeeze()

#create matrix for input and targets
#the inputs are the first n values in the input data
#the target is immediate value after that in the input data

#temp vars
input_data_len = input_data.size

#input and target numpy martracies
input_numpy = numpy.zeros([input_data_len - d, d])
target_numpy = numpy.zeros([input_data_len - d])

for i in range(input_data_len - d):
    input_numpy[i, :] = input_data[i:i+d]
    target_numpy[i] = input_data[i+d]

data_tensor = torch.tensor(input_numpy, dtype=torch.float32)
target_tensor = torch.tensor(target_numpy, dtype=torch.float32)

target_tensor = torch.unsqueeze(target_tensor, dim=1)

model = SequencePredictor(num_features=d)
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

#predict some values 
if use_gpu and torch.cuda.device_count() >= 1:
    data = data_tensor.cpu().detach().numpy()
    predicted_targets = model(data_tensor).cpu()
else:
    data = data_tensor.detach().numpy()
    predicted_targets = model(data_tensor)

predicted_targets = model(data_tensor)

#print(torch.round((target_tensor - predicted_targets), decimals=1))

possible_values = numpy.zeros(10)

for i in range(10):
    possible_values[i] = i

possible_values_tensor = torch.tensor(possible_values, dtype=torch.float32)
possible_values_tensor = possible_values_tensor.unsqueeze(1)

if use_gpu:
    if torch.cuda.device_count() >= 1:
        possible_values_tensor = possible_values_tensor.to(torch.device(f'cuda:{0}'))

print(possible_values_tensor)
predicted_targets = model(possible_values_tensor)

print(predicted_targets)

#print(torch.round((target_tensor - predicted_targets), decimals=1))
'''
color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])#color_map[data[:,n]])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())#color_map[predicted_classes])
plot.show()'''
