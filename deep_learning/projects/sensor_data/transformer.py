#traditional transformer architecture

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import numpy 
import pandas
import os
import math
import matplotlib.pyplot as pyplot 
import matplotlib
from progress.bar import Bar as ProgressBar

#Data
class SensorDataset(Dataset):
    def __init__(self, input_depth=1, pred_depth=1, use_gpu=False):
        self.use_gpu = use_gpu
        self.input_depth = input_depth
        self.pred_depth = pred_depth 

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
        return self.input_data.size(dim=0) - 1 - self.input_depth - self.pred_depth

    def __getitem__(self, ind):
        #Array of inputs, output
        #return ([self.input_data[i, :] for i in range(ind, self.input_depth)], self.input_data[ind + self.input_depth, :])
        return self.input_data[ind: ind + self.input_depth, :], self.input_data[ind + input_depth: ind + self.input_depth + self.pred_depth, :]

class TimeSeriesEncoder(torch.nn.Module):
    def __init__(self, num_hiddens, num_heads, input_depth, use_gpu=False):
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(embed_dim=3, num_heads=num_heads, batch_first=True, dtype=torch.float64)
        self.norm_1 = torch.nn.LayerNorm(normalized_shape=(input_depth, 3), dtype=torch.float64)
        self.linear_1 = torch.nn.Linear(in_features=3, out_features=num_hiddens, dtype=torch.float64)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(in_features=num_hiddens, out_features=3, dtype=torch.float64)
        self.norm_2 = torch.nn.LayerNorm(normalized_shape=(input_depth, 3), dtype=torch.float64)

        if use_gpu:
            if torch.cuda.device_count() >= 1:
                self.attention_1.to(torch.device(f'cuda:{0}'))
                self.norm_1.to(torch.device(f'cuda:{0}'))
                self.linear_1.to(torch.device(f'cuda:{0}'))
                self.relu_1.to(torch.device(f'cuda:{0}'))
                self.linear_2.to(torch.device(f'cuda:{0}'))
                self.norm_2.to(torch.device(f'cuda:{0}'))

    def forward(self, X):
        #multihead attention
        output, _ = self.attention(X, X, X)
        #add and normalize
        X_2 = self.norm_1(output + X)
        #positionwise Linear network
        output = self.linear_2(self.relu_1(self.linear_1(X_2)))
        #add and normalize
        output = self.norm_2(output + X_2)

        return output

class TimeSeriesDecoder(torch.nn.Module):
    def __init__(self, num_hiddens, num_heads, pred_depth, use_gpu=False):
        super().__init__()

        self.pred_depth = pred_depth

        self.attention_1 = torch.nn.MultiheadAttention(embed_dim=3, num_heads=num_heads, batch_first=True, dtype=torch.float64)
        self.norm_1 = torch.nn.LayerNorm(normalized_shape=(pred_depth, 3), dtype=torch.float64)
        self.attention_2 = torch.nn.MultiheadAttention(embed_dim=3, num_heads=num_heads, batch_first=True, dtype=torch.float64)
        self.norm_2 = torch.nn.LayerNorm(normalized_shape=(pred_depth, 3), dtype=torch.float64)
        self.linear_1 = torch.nn.Linear(in_features=3, out_features=num_hiddens, dtype=torch.float64)
        self.relu_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(in_features=num_hiddens, out_features=3, dtype=torch.float64)
        self.norm_3 = torch.nn.LayerNorm(normalized_shape=(pred_depth, 3), dtype=torch.float64)

        # Set self attention mask
        self.self_attention_mask = torch.tril(torch.ones(pred_depth, pred_depth))

        if use_gpu:
            if torch.cuda.device_count() >= 1:
                self.attention_1.to(torch.device(f'cuda:{0}'))
                self.norm_1.to(torch.device(f'cuda:{0}'))
                self.attention_2.to(torch.device(f'cuda:{0}'))
                self.norm_2.to(torch.device(f'cuda:{0}'))
                self.linear_1.to(torch.device(f'cuda:{0}'))
                self.relu_1.to(torch.device(f'cuda:{0}'))
                self.linear_2.to(torch.device(f'cuda:{0}'))
                self.norm_3.to(torch.device(f'cuda:{0}'))

    def forward(self, X, encoder_output):
        #multihead attention
        output, _ = self.attention_1(query=X, key=X, value=X, attn_mask=self.self_attention_mask)
        #add and normalize
        X_2 = self.norm_1(output + X)
        #multihead attention
        output, _ = self.attention_2(query=X_2, key=encoder_output, value=encoder_output)
        #add and normalize
        X_3 = self.norm_1(output + X_2)
        #positionwise Linear network
        output = self.linear_2(self.relu_1(self.linear_1(X_3)))
        #add and normalize
        output = self.norm_3(output + X_3)

        return output

    def zero_input(self):
        return torch.zeros((self.pred_depth, 3), dtype=torch.float64)

class TimeSeriesSensorPredictor(torch.nn.Module):
    def __init__(self, num_hiddens, num_heads, input_depth, pred_depth, use_gpu=False):
        super().__init__()

        self.pred_depth = pred_depth

        self.encoder = TimeSeriesEncoder(num_hiddens, num_heads, input_depth, use_gpu)
        self.decoder = TimeSeriesDecoder(num_hiddens, num_heads, pred_depth, use_gpu)
        self.position_encoder = Summer(PositionalEncoding1D(input_depth))

    def forward(self, X, T):
        #Position Encode
        p_X = self.position_encoder(X)
        p_T = self.position_encoder(T)

        #input 
        output = self.encoder(p_X)
        output = self.decoder(p_T, output)

        return output

    def predict(self, X):
        #Position Encode
        p_X = self.position_encoder(X)

        #pass to encoder output
        enc_output = self.encoder(p_X)

        #get initial zero input (meaning we have no predictions yet)
        dec_input = self.decoder.zero_input().unsqueeze(0)

        for i in range(self.pred_depth):
            #position encode
            p_dec_input = self.position_encoder(dec_input)

            #mask all values, and replace with predicted values
            dec_output = self.decoder(p_dec_input, enc_output)

            #replace predicted output in proper position 
            dec_input[:, i, :] = dec_output[:, i, :]

        return dec_input


#Model Variables
num_epochs = 20 
learning_rate = 1.0e-02 
training_data_batch_size = 4 
# How many sequential inputs to use in prediction
input_depth = 10 
# How many values to predict
predict_depth = 3
use_gpu = False 

dataset = SensorDataset(input_depth=input_depth, pred_depth=predict_depth, use_gpu=True)
generator = torch.Generator().manual_seed(10)
training_data_length = math.floor(0.8 * len(dataset)) 
training_dataset, test_dataset = random_split(dataset, [training_data_length, len(dataset) - training_data_length], generator=generator) 

#Generate data loaders for data
training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=training_data_batch_size, num_workers=0)
test_data_loader = DataLoader(test_dataset, shuffle=True, num_workers=0)

#Components
model = TimeSeriesSensorPredictor(2, 3, input_depth, predict_depth)
optimizer = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=learning_rate
)
loss_function = torch.nn.L1Loss()

#Dynamic Graph
store_losses = []

#Progress Bar
progress_bar = ProgressBar('processing', max=num_epochs * (len(training_dataset) / training_data_batch_size))

for epoch in range(num_epochs):
    avg_loss = 0

    #Move forward
    for i, (inputs, target) in enumerate(training_data_loader):
        #Get predictions
        #Get last output of each RNN iteration to use as predition
        y_pred = model(inputs, target)

        #Compute loss
        #Only compute loss over final output, and not all outputs
        loss = loss_function(y_pred, target)

        optimizer.zero_grad()

        #Compute gradients
        loss.backward()

        #Clip gradients
        #torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)

        #increment weights
        optimizer.step()

        #Compute batch loss for graph
        store_losses += [(loss / (target.sum() / input_depth / training_data_batch_size)).item()]

        progress_bar.next()

progress_bar.finish()

# Compute Loss over training data
#TODO predict does not support batched data
#training_sample_inputs, training_sample_targets = next(iter(training_data_loader))
#sample_loss = loss_function(model.predict(training_sample_inputs), training_sample_targets).detach().item()

# Compute Loss over test data
avg_loss = 1

for i, (inputs, targets) in enumerate(test_data_loader):
    prediction = model.predict(inputs) 
    print('predition was {pred} and target was {target}'.format(pred=prediction, target=targets))
    avg_loss = avg_loss + loss_function(prediction, targets).detach().item()

    #TODO DELETE
    if i == 1:
        break

#avg_loss = avg_loss / len(test_dataset)

print("model has " + str(sum([param.nelement() for param in model.parameters()])) + " parameters")
#print(f"sample loss was {sample_loss}")
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