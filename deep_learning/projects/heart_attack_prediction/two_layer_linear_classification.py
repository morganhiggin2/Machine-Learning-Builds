import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy 

class PredictionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        linear_one = torch.nn.Linear(13, 5)
        tanh = torch.nn.Tanh()
        linear_two = torch.nn.Linear(5, 2)
        self.net = torch.nn.Sequential(linear_one, tanh, linear_two)

    def forward(self, X):
        return self.net(X)

#Data
class HeartAttackDataset(Dataset):
    def __init__(self):
        raw_data = numpy.genfromtxt("../data/heart_attack_prediction/heart.csv", delimiter=",", skip_header=1)
        self.input = torch.from_numpy(raw_data[:, list(range(0, raw_data.shape[1] - 1))])
        self.input = self.input.type(torch.FloatTensor)

        #turn into 2d with class labels
        output = torch.from_numpy(raw_data[:, raw_data.shape[1] - 1])
        output = output.type(torch.int64)
        self.output = torch.nn.functional.one_hot(output, num_classes=2).type(torch.FloatTensor)
    
    def __len__(self):
        return len(self.input) 

    def __getitem__(self, ind):
        return self.input[ind], self.output[ind]

dataset = HeartAttackDataset()

#Model Variables
num_epochs = 50
learning_rate = 1.0e2 
training_data_batch_size = 3

#Split data into training and test data with random generator
generator = torch.Generator().manual_seed(10)
training_data, test_data = random_split(dataset, [250, 53], generator=generator) 

#Generate data loaders for data
training_data_loader = DataLoader(training_data, batch_size=training_data_batch_size, shuffle=True, num_workers=0)
test_data_loader = DataLoader(test_data, shuffle=True, num_workers=0)

#Components
model = PredictionNet()
optimizer = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=learning_rate
)
loss_function = torch.nn.CrossEntropyLoss()

for i in range(num_epochs):
    #Move forward
    for i, (inputs, targets) in enumerate(training_data_loader):
        #Get predictions
        y_pred = model.forward(inputs)

        #Compute Loss
        loss = loss_function(y_pred, targets)

        optimizer.zero_grad()

        #Compute Gradients, increment weight
        loss.backward()
        optimizer.step()

# Compute Loss over traning data
training_sample_inputs, training_sample_targets = next(iter(training_data_loader))
sample_loss = loss_function(model(training_sample_inputs), training_sample_targets) / training_data_batch_size

# Compute Loss over test data
avg_loss = 0

for inputs, targets in training_data_loader:
    avg_loss = avg_loss + loss_function(model(inputs), targets)

avg_loss = avg_loss / len(training_data)

print(f"sample loss was {sample_loss}")
print(f"traning data loss was {avg_loss}")
print(str(loss_function.weight))