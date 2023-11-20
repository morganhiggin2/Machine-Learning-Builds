import torch
import torch.nn
import pandas
import numpy
import random
import sklearn.datasets
import matplotlib.pyplot as plot
from matplotlib import colors

class SequencePredictor(torch.nn.Module):
    def __init__(self, input_size, output_size, num_hidden, num_layers):
        super().__init__()

        self.hidden_dim = num_hidden
        self.num_layers = num_layers

        self.rec = torch.nn.RNN(input_size=input_size, hidden_size=num_hidden, nonlinearity='relu')
        self.net = self.rec
        #self.lin = torch.nn.Linear(in_features=num_hidden, out_features=output_size)

        #self.net = torch.nn.Sequential(self.rec, slf.lin)

    def init_hidden(self, batch_size):
        hidden = None

        if batch_size == 0:
            hidden = torch.zeros(self.num_layers, self.hidden_dim)
        else:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        if use_gpu:
            if torch.cuda.device_count() >= 1:
                hidden = hidden.to(torch.device(f'cuda:{0}'))

        return hidden

    def forward(self, X):
        #batch_size = X.size(0)

        hidden = self.init_hidden(batch_size=0)

        output, hidden = self.net(X, hidden)
        
        '''output = out.contiguous().view(-1, self.hidden_dim)
        output = self.lin(output)'''

        return output, hidden
        
n = 3
d = 2
h = n + 1
clip_max = 1.0
e_loss = 1e-5
use_gpu = True
file_name = "data/simple_seq_target_1_data.csv"
num_data = 50000

#generate sequencial 1d data based on a function
#generate patterns with target values, then have others be random
patterns = [([1,3,2], 0.1), ([3,5,4], 0.2)]

data = []
targets = []
pattern_targets = []

for i in range(num_data):
    noise = random.gauss(mu=0.0, sigma=0.1) 

    if random.random() < 0.8:
        data.append(random.randint(0,9))
        targets.append(random.random())
        pattern_targets.append(False)
    else:
        pat_index = random.randint(0, len(patterns) - 1)
        data.extend(patterns[pat_index][0])
        targets.extend([random.random() for i in range(len(patterns[pat_index][0]) - 1)])
        targets.append(patterns[pat_index][1])
        pattern_targets.extend([False for i in range(len(patterns[pat_index][0]) - 1)])
        pattern_targets.append(True)

data_tensor = torch.tensor(data, dtype=torch.float32)
target_tensor = torch.tensor(targets, dtype=torch.float32)

target_tensor = torch.unsqueeze(target_tensor, dim=1)
data_tensor = torch.unsqueeze(data_tensor, dim=1)

#create the model and train data
model = SequencePredictor(input_size=data_tensor.size(1), output_size=1, num_hidden=1, num_layers=1)
optim = torch.optim.Adam(
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
    output, hidden = model(data_tensor)
    
    loss = loss_fn(output, target_tensor)
   
    optim.zero_grad()
   
    loss.backward()

    #torch.nn.utils.clip_grad_norm(model.parameters(), clip_max)

    optim.step()

    if epoch % 10 == 0:
        print(str(epoch) + " " + str(loss))

    curr_loss = loss.detach().item()

    if abs(curr_loss - prev_loss) < e_loss:
        break

    prev_loss = curr_loss

if use_gpu and torch.cuda.device_count() >= 1:
    data = data_tensor.cpu().detach().numpy()
else:
    data = data_tensor.detach().numpy()

predicted_targets = model(data_tensor)[0]

#print(torch.round((target_tensor - predicted_targets), decimals=1))
print(torch.abs(target_tensor - predicted_targets).mean())

#analyize the average value of the pattern targets
total = 0.0
n = 0 

for i in range(len(pattern_targets)):
    if pattern_targets[i]:
        total += abs((target_tensor[i] - predicted_targets[i]).cpu().detach())
        n += 1 

print(total / n)
'''
color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])#color_map[data[:,n]])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())#color_map[predicted_classes])
plot.show()'''
