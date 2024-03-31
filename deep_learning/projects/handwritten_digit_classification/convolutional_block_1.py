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
class HandwritingDataset(Dataset):
    def __init__(self, test=False, use_gpu=False):
        sub_dir = "test" if test else "train"
        image_csv_file = "../data/handwritten_digit_classification/" + sub_dir + '.csv'
        self.class_labels = numpy.arange(0, 9).astype(dtype=str) 
        self.use_gpu = use_gpu

        #Becuase the dataset is small enough to keep in memory, we will load it at once and just pull it
        #this will be a list of tensors
        image_csv = pandas.read_csv(image_csv_file, index_col=False, dtype=numpy.float32)

        self.images = []
        self.labels = []
        for i, row in image_csv.iterrows():
            if sub_dir == "train":
                image = torch.from_numpy(row.iloc[1:785].to_numpy())
                digit_class = torch.nn.functional.one_hot(torch.tensor(row['label'].astype(dtype=numpy.int64)), num_classes=10)
                digit_class = digit_class.float()
            else:
                image = torch.from_numpy(row.iloc[0:784].to_numpy())
                digit_class = torch.zeros(10, dtype=torch.float32) 

            image = image.reshape(shape=(28, 28))
            #TODO check if this is messing up our convergence at larger batch sizes
            image = image.unsqueeze(0)

            if self.use_gpu:
                if torch.cuda.device_count() >= 1:
                    image.to(torch.device(f'cuda:{0}'))
                    digit_class.to(torch.device(f'cuda:{0}'))

            self.images += [image]
            self.labels += [digit_class]

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, ind):
        return self.images[ind], self.labels[ind]

class HandwrittenClassifier(torch.nn.Module):
    def __init__(self, use_gpu=False):
        super().__init__()

        image_dimensions = (28, 28) 
 
        conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=6, stride=1, padding='same')
        relu_1 = torch.nn.ReLU()
        max_pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        conv_2_1 = torch.nn.Conv2d(in_channels=6, out_channels=20, kernel_size=3, stride=1, padding='same')
        conv_2_2= torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding='same')
        conv_2_3 = torch.nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding='same')
        relu_2 = torch.nn.ReLU()
        max_pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        flatten_1 = torch.nn.Flatten(1, 3)
        linear_1 = torch.nn.Linear(in_features=490, out_features=10)
        relu_3 = torch.nn.ReLU()
        self.net = torch.nn.Sequential(conv_1, relu_1, max_pool_1, conv_2_1, conv_2_2, conv_2_3, relu_2, max_pool_2, flatten_1, linear_1, relu_3)

        if use_gpu:
            if torch.cuda.device_count() >= 1:
                self.net.to(torch.device(f'cuda:{0}'))

    def forward(self, X):
        return self.net(X)
#Model Variables
num_epochs = 8 
learning_rate = 1.0e-3 
training_data_batch_size = 40 
use_gpu = True 

training_dataset = HandwritingDataset(use_gpu=True)
test_dataset = HandwritingDataset(test=True, use_gpu=True)

#Generate data loaders for data
training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=training_data_batch_size, num_workers=0)
test_data_loader = DataLoader(test_dataset, shuffle=True, num_workers=0)

#Components
model = HandwrittenClassifier()
optimizer = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=learning_rate
)
loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

#Dynamic Graph
store_losses = []

#Progress Bar
progress_bar = ProgressBar('processing', max=num_epochs * (len(training_dataset) / training_data_batch_size))

#Show how many parameters there are:

for epoch in range(num_epochs):
    avg_loss = 0

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

        #Compute batch loss for graph
        store_losses += [loss.item()]

        progress_bar.next()

progress_bar.finish()

# Compute Loss over training data
training_sample_inputs, training_sample_targets = next(iter(training_data_loader))
sample_loss = loss_function(model(training_sample_inputs), training_sample_targets).detach().item()

# Compute Loss over test data
avg_loss = 0
n = 0

for i, (inputs, targets) in enumerate(test_data_loader):
    avg_loss = avg_loss + loss_function(model(inputs), targets).detach().item()
    n += 1

avg_loss = avg_loss / n 

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