#seperate into 3 channels
#convolutional of k x k
#max pooling
#flatten amoung channels
#2 layer linear with relu on first, softmax into c classes on second

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
import numpy 
import pandas
import os
import math
import matplotlib.pyplot as pyplot 
import matplotlib
from progress.bar import Bar as ProgressBar

#Data
class SensorDataset(Dataset):
    def __init__(self, hidden_depth=1, use_gpu=False):
        self.use_gpu = use_gpu
        self.hidden_depth = hidden_depth
        
        #Load data from csv
        dataframe = pandas.read_csv('../data/sensor_data/sensor_data.csv', dtype=str)

        #Fill null values with 0
        dataframe = dataframe.fillna(0.0)

        #Convert columns into their numpy float 64 counterpart
        dataframe['SensorA'] = dataframe['SensorA'].astype(dtype=numpy.float64)
        dataframe['SensorB'] = dataframe['SensorB'].astype(dtype=numpy.float64)
        dataframe['SensorC'] = dataframe['SensorC'].astype(dtype=numpy.float64)
        
        #Drop the time column, no necessary
        dataframe = dataframe.drop(columns=['time'])

        #Convert dataframe into numpy
        numpy_data = dataframe.to_numpy(dtype=numpy.float64)

        #prepare into time series data
        #just do tensor for now and use slices
        self.input_data = torch.from_numpy(numpy_data)
        self.input_data.requires_grad = True

        if self.use_gpu:
            if torch.cuda.device_count() >= 1:
                self.input_data.to(torch.device(f'cuda:{0}'))

    def __len__(self):
        return self.input_data.size(dim=0) - 1 - self.hidden_depth 

    def __getitem__(self, ind):
        #Array of inputs, output
        #return ([self.input_data[i, :] for i in range(ind, self.hidden_depth)], self.input_data[ind + self.hidden_depth, :])
        return self.input_data[ind: ind + self.hidden_depth, :], self.input_data[ind + self.hidden_depth, :].unsqueeze(0)

class TimeSeriesSensorPredictor(torch.nn.Module):
    def __init__(self, hidden_depth=1, use_gpu=False):
        super().__init__()

        self.num_hidden_features = 3
        self.net = torch.nn.RNN(input_size=3, hidden_size=self.num_hidden_features, num_layers=1, batch_first=True, dtype=torch.float64, nonlinearity='relu') 
        self.hidden_depth = hidden_depth

        if use_gpu:
            if torch.cuda.device_count() >= 1:
                self.net.to(torch.device(f'cuda:{0}'))

    def forward(self, X, H):
        return self.net(X, H)

    def get_zero_hidden(self, num_batches):
        return torch.zeros((1, num_batches, self.num_hidden_features), dtype=torch.float64)

#Model Variables
num_epochs = 3 
learning_rate = 1.0e-04 
training_data_batch_size = 10 
hidden_depth = 2 
use_gpu = True 

dataset = SensorDataset(hidden_depth=hidden_depth, use_gpu=True)
generator = torch.Generator().manual_seed(10)
training_data_length = math.floor(0.8 * len(dataset)) 
training_dataset, test_dataset = random_split(dataset, [training_data_length, len(dataset) - training_data_length], generator=generator) 

#Generate data loaders for data
training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=training_data_batch_size, num_workers=0)
test_data_loader = DataLoader(test_dataset, shuffle=True, num_workers=0)

#Components
model = TimeSeriesSensorPredictor()
optimizer = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=learning_rate
)
loss_function = torch.nn.L1Loss()

#Dynamic Graph
store_losses = []

#Progress Bar
progress_bar = ProgressBar('processing', max=num_epochs * (len(training_dataset) / training_data_batch_size))

def evaluate_sequence(inputs, batch_size): 
    hidden = torch.tensor(model.get_zero_hidden(inputs.size(0)), requires_grad=True) 
    y_pred, _hidden = model(inputs, hidden)

    return y_pred[:, hidden_depth - 1, :]


for epoch in range(num_epochs):
    avg_loss = 0

    #Move forward
    for i, (inputs, target) in enumerate(training_data_loader):
        #Get predictions
        #Get last output of each RNN iteration to use as predition
        y_pred = evaluate_sequence(inputs, inputs.size(0))

        #Compute Loss
        #Only compute loss over final output, and not all outputs
        loss = loss_function(y_pred, target)

        optimizer.zero_grad()

        #Compute Gradients, increment weight
        loss.backward()
        optimizer.step()

        #Compute batch loss for graph
        store_losses += [loss.item()]

        progress_bar.next()

progress_bar.finish()

# Compute Loss over training data
training_sample_inputs, training_sample_targets = next(iter(training_data_loader))
sample_loss = loss_function(evaluate_sequence(training_sample_inputs, training_data_batch_size), training_sample_targets).detach().item()

# Compute Loss over test data
avg_loss = 1

for i, (inputs, targets) in enumerate(test_data_loader):
    prediction = evaluate_sequence(inputs, 1) 
    print('predition was {pred} and target was {target}'.format(pred=prediction, target=targets))
    avg_loss = avg_loss + loss_function(prediction, targets).detach().item()

    #TODO DELETE
    if i == 1:
        break

avg_loss = avg_loss / len(test_dataset)

'''
# Compute Loss over training data
training_sample_inputs, training_sample_targets = next(iter(training_data_loader))
hidden = torch.clone(model.get_zero_hidden(training_data_batch_size)) 
sample_loss = loss_function(model(training_sample_inputs, hidden)[0][:, hidden_depth - 1, :], training_sample_targets).detach().item()

# Compute Loss over test data
avg_loss = 1

for i, (inputs, targets) in enumerate(test_data_loader):
    hidden = torch.clone(model.get_zero_hidden(1)) 
    prediction = model(inputs, hidden)[0][:, hidden_depth - 1, :]
    print('predition was {pred} and target was {target}'.format(pred=prediction, target=targets))
    avg_loss = avg_loss + loss_function(prediction, targets).detach().item()

    #TODO DELETE
    if i == 1:
        break

avg_loss = avg_loss / len(test_dataset)
'''

print("model has " + str(sum([param.nelement() for param in model.parameters()])) + " parameters")
print(f"sample loss was {sample_loss}")
print(f"test data loss was {avg_loss}")

#Plot with matplotlib
pyplot.style.use('ggplot')
matplotlib.use('TkAgg')           

step_size = 1.0 / math.ceil(len(training_dataset) / training_data_batch_size)
graph_epochs = numpy.arange(0, len(store_losses)) * step_size
graph_losses = numpy.array(store_losses)


figure = pyplot.figure()
axis = figure.add_subplot(111)
line, = axis.plot(graph_epochs, graph_losses, 'o')
axis.set_xlabel('epoch')
axis.set_ylabel('loss')

pyplot.show()
#watch -n 0.5 nvidia-smi