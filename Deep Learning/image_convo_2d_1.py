import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.utils as utils
import PIL
import pandas
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plot
from matplotlib import colors

class ConvoClassifier(torch.nn.Module):
    def __init__(self, num_channels, kern_d1, kern_d2):
        super().__init__()

        self.convo = torch.nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=(kern_d1, kern_d2), padding='same', padding_mode='zeros')

        self.net = torch.nn.Sequential(self.convo)

    def forward(self, X):
        return self.net(X)

def getLineNumberOfString(string, file_name):
    file = open(file_name)
    
    for num, line in enumerate(file, 1):
        if string in line:
            return num
    return -1

#import image as tensor
image_data = PIL.Image.open('data/image_similes_data_1.jpg').convert('RGB')
image_targets = PIL.Image.open('data/image_similes_target_1.png').convert('RGB')

transform = transforms.Compose([
    transforms.PILToTensor()
])

data_tensor = transform(image_data)
target_tensor = transform(image_targets)

data_tensor = data_tensor.float()
target_tensor = target_tensor.float()[0]

target_tensor = target_tensor.reshape((1, target_tensor.size(0), target_tensor.size(1)))

max_target_value = torch.max(target_tensor)

target_tensor[:] = target_tensor / max_target_value
data_tensor[:] = data_tensor / 256

#define variables for convolution neural network
n = 2
h = n + 1
e_loss = 0.00001
use_gpu = True
file_name = "data/convo_plus_minus_grid.csv"

model = ConvoClassifier(num_channels=data_tensor.size(0), kern_d1=6, kern_d2=5)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-3
)
loss_fn = torch.nn.L1Loss()

target_tensor[:] = target_tensor

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
    #loss = torch.abs(output - target_tensor).sum() 
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

transform = transforms.Compose([
    transforms.ToPILImage(mode='RGB')
])

max_val = abs(torch.max(predicted_targets))
predicted_targets[:] = predicted_targets * (1.0 / max_val)

pandas_predicted_values = pandas.DataFrame(predicted_targets[0].cpu().detach().numpy())
pandas_predicted_values.to_csv("data/image_convo_2d_1_output.csv")

predicted_targets_rgb = torch.zeros((3, predicted_targets.size(1), predicted_targets.size(2)), dtype=torch.uint8)
#predicted_targets_rgb[0] = 
pred_temp = torch.clamp(torch.abs(predicted_targets[0] * 256), min=0.0, max=256.0).type(torch.uint8)
predicted_targets_rgb[0] = pred_temp

#utils.save_image(torch.abs(predicted_targets_rgb), fp="data/image_convo_2d_1_output.jpg")

predicted_image = transform(predicted_targets_rgb)

predicted_image.save("data/image_convo_2d_1_output.png")

'''
color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])#color_map[data[:,n]])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())#color_map[predicted_classes])
plot.show()'''
