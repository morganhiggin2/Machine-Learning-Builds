import torch
import torch.nn
import pandas
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plot
from matplotlib import colors

class ConvoClassifier(torch.nn.Module):
    def __init__(self, num_channels, kern_d1, kern_d2):
        super().__init__()

        self.convo = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(kern_d1, kern_d2), padding='same', padding_mode='zeros')

        self.net = torch.nn.Sequential(self.convo)

    def forward(self, X):
        return self.net(X)

def getLineNumberOfString(string, file_name):
    file = open(file_name)
    
    for num, line in enumerate(file, 1):
        if string in line:
            return num
    return -1

n = 2
h = n + 1
e_loss = 0.0001
use_gpu = True
file_name = "data/convo_plus_minus_grid.csv"

table_names = ["Data", "Target"]
table_name_rows = [getLineNumberOfString(string, file_name) for string in table_names]

raw_data = pandas.read_csv(file_name, index_col=False, header=None, skiprows=table_name_rows[0], nrows=table_name_rows[1] - 2)
raw_targets = pandas.read_csv(file_name, index_col=False, header=None, skiprows=table_name_rows[1]-table_name_rows[0] + 1)

data_tensor = torch.tensor(raw_data.values, dtype=torch.float32)
target_tensor = torch.tensor(raw_targets.values, dtype=torch.float32)

data_tensor = data_tensor.reshape((1, 1, data_tensor.size(0), data_tensor.size(1)))
target_tensor = target_tensor.reshape((1, 1, target_tensor.size(0), target_tensor.size(1)))

model = ConvoClassifier(num_channels=1, kern_d1=3, kern_d2=3)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-2
)
loss_fn = torch.nn.MSELoss()

target_tensor[:] = target_tensor - torch.ones(target_tensor.shape)

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

    if epoch % 1 == 0:
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
