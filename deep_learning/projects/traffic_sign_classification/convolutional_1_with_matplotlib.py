#WARNING MATPLOTLIB DOES NOT WORK

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

#Data
class TrafficSignDataset(Dataset):
    def __init__(self, test=False, use_gpu=False):
        sub_dir = "test" if test else "train"
        self.image_directory = "../data/traffic_sign_classification/" + sub_dir + "/images/"
        self.labels_directory = "../data/traffic_sign_classification/" + sub_dir + "/labels/"
        self.class_labels = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']
        self.image_dimensions = (416, 416)
        self.use_gpu = use_gpu

        #get image labels
        column_names = ["id", "class_label_index", "x1", "y1"]

        #read all files in images directory
        directory_list = os.listdir(self.image_directory) 

        #only handling jpg (which should be the only data type)
        image_ids = set(map(lambda image_name: image_name[0:-4], filter(lambda file_name: True if file_name[-4:] == '.jpg' else False, directory_list)))

        valid_image_ids = set()
        #filter out all image ids for which labels don't exist
        for image_id in image_ids:
            if os.path.isfile(self.labels_directory + image_id + ".txt"):
                valid_image_ids.add(image_id)

        #create dataframe with ids, and empty columns for label data
        self.image_registry = pandas.DataFrame(data=valid_image_ids, columns=['id'])
        
        #create random assignment of ids
        self.id_ref = numpy.arange(len(self.image_registry))
        numpy.random.shuffle(self.id_ref)
    
    def __len__(self):
        return len(self.image_registry) 

    def __getitem__(self, ind):
        #index lookup 
        ind_ref = self.id_ref[ind]

        #find row in image labels set by row index
        image_label = self.image_registry.iloc[ind_ref]['id']

        #with image id from row, retrive image
        image_tensor = read_image(self.image_directory + image_label + '.jpg')
        #normalize
        image_tensor = torch.nn.functional.normalize(image_tensor.float(), dim=1)

        #get label data from file
        file_data = numpy.genfromtxt(self.labels_directory + image_label + '.txt', usemask=True) 

        if len(file_data) == 0:
            file_data = numpy.zeros((1, 15))

        if file_data.ndim == 1:
            file_data = numpy.expand_dims(file_data, axis=0)

        binary_class_labels = torch.nn.functional.one_hot(torch.from_numpy(file_data[:,0].astype(dtype=numpy.int64)), num_classes=len(self.class_labels))

        # Because an image can have multiple classes, and each tensor for each target must have the same size, we are going to marge the binary class labels
        # into one (ex. if an image is two classes, then it will be 0.5 of one class and 0.5 of the other).

        # Merge binary class labels
        binary_class_label = torch.sum(binary_class_labels, dim=0)
        binary_class_label = binary_class_label.type(dtype=torch.float64)
        
        # Normalize
        #binary_class_label = torch.nn.functional.normalize(binary_class_label, dim=0)
        #binary_class_label = torch.nn.functional.normalize(binary_class_label, dim=0)

        #put important information (all columns except image id and row index)
        target_tensor = binary_class_label

        if self.use_gpu:
            if torch.cuda.device_count() >= 1:
                image_tensor.to(torch.device(f'cuda:{0}'))
                target_tensor.to(torch.device(f'cuda:{0}'))

        return (image_tensor, target_tensor)

class TrafficSignClassifier(torch.nn.Module):
    def __init__(self, use_gpu=False):
        super().__init__()

        image_dimensions = (416, 416)

        conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=11, stride=7, padding=2)
        relu_1 = torch.nn.ReLU()
        max_pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        conv_2 = torch.nn.Conv2d(in_channels=6, out_channels=15, kernel_size=5, stride=2, padding=2)
        relu_2 = torch.nn.ReLU()
        max_pool_2 = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        flatten_1 = torch.nn.Flatten(1, 3)
        linear_1 = torch.nn.Linear(in_features=135, out_features=96)
        relu_3 = torch.nn.ReLU()
        linear_2 = torch.nn.Linear(in_features=96, out_features=15)
        relu_4 = torch.nn.ReLU()
 
        self.net = torch.nn.Sequential(conv_1, relu_1, max_pool_1, conv_2, relu_2, max_pool_2, flatten_1, linear_1, relu_3, linear_2, relu_4)

        if use_gpu:
            if torch.cuda.device_count() >= 1:
                self.net.to(torch.device(f'cuda:{0}'))

    def forward(self, X):
        return self.net(X)

#Model Variables
num_epochs = 50
learning_rate = 2.0e-3 
training_data_batch_size = 40 
use_gpu = True

traning_dataset = TrafficSignDataset(use_gpu=True)
test_dataset = TrafficSignDataset(test=True, use_gpu=True)

#Generate data loaders for data
training_data_loader = DataLoader(traning_dataset, shuffle=True, batch_size=training_data_batch_size, num_workers=0)
test_data_loader = DataLoader(test_dataset, shuffle=True, num_workers=0)

#Components
model = TrafficSignClassifier()
optimizer = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=learning_rate
)
loss_function = torch.nn.CrossEntropyLoss()

#Dynamic Graph
store_losses = [0, 1]
graph_epochs = numpy.linspace(0, 1, 2, dtype=numpy.int32)
graph_losses = numpy.array(store_losses, dtype=numpy.int32)
pyplot.style.use('ggplot')
matplotlib.use('TkAgg')
figure = pyplot.figure()
axis = figure.add_subplot(111)
line, = axis.plot(graph_epochs, graph_losses, 'o')

figure.canvas.draw()
pyplot.show(block=False)

#graph.title('losses')
#graph.set_xlabel('epoch')
#graph.set_ylabel('loss')

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

        if i % 8 == 0 and i > 0 or epoch > 0:
            step_size = 1.0 / math.ceil(len(traning_dataset) / training_data_batch_size)
            graph_epochs = numpy.arange(0, step_size * (len(store_losses)), step_size)
            graph_losses = numpy.array(store_losses)

            line.set_data(graph_epochs, graph_losses)

            axis.autoscale_view(True, True, True)
            axis.relim()

            figure.canvas.draw()
 

# Compute Loss over traning data
training_sample_inputs, training_sample_targets = next(iter(training_data_loader))
sample_loss = loss_function(model(training_sample_inputs), training_sample_targets).detach().item()

# Compute Loss over test data
avg_loss = 0

for i, (inputs, targets) in enumerate(test_data_loader):
    avg_loss = avg_loss + loss_function(model(inputs), targets).detach().item()

avg_loss = avg_loss / len(test_dataset)

print(f"sample loss was {sample_loss}")
print(f"test data loss was {avg_loss}")

#watch -n 0.5 nvidia-smi