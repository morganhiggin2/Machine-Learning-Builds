import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import colors
import torch_geometric
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

class GCNConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bais.data.zero_()

    def forward(self, x, edge_index):
        #add self loops 
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        #linearly transform node feature matrix
        x = self.lin(x)

        #compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        #start propragating messages 
        out = self.propagate(edge_index, x=x, norm=norm)

        #apply a final bias vector
        out += self.bias

        return out
    
    def message(self, x_j, norm):
        #normalize node features
        return norm.view(-1, 1) * x_j

#create the graph 
#    0 ------ 1 ------ 2 ------ 3 
#    |-----------------|
#             | --------------- |

e_loss = 0.001
use_gpu = True

edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]], dtype=torch.long).t().contiguous()
node_values = torch.tensor([[0], [0.5], [0.1], [0.4]])
data_tensor = Data(x=node_values, edge_index=edges)

target_tensor = torch.tensor([[1], [0.5], [0.3], [0.1]])

model = GCNConv(in_channels=1, out_channels=1)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-2
)
loss_fn = torch.nn.MSELoss()

target_tensor[:] = target_tensor - torch.ones(target_tensor.shape)

data_tensor.requires_grad_(True)
target_tensor.requires_grad_(True)

print(use_gpu)

if use_gpu:
    if torch.cuda.device_count() >= 1:
        model.net = model.net.to(torch.device(f'cuda:{0}'))
        data_tensor = data_tensor.to(torch.device(f'cuda:{0}'))
        target_tensor = target_tensor.to(torch.device(f'cuda:{0}'))

prev_loss = -1
curr_loss = -1

for epoch in range(500):
    output = model(x=data_tensor.x, edge_index=data_tensor.edge_index)
    
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

predicted_targets = None

if use_gpu and torch.cuda.device_count() >= 1:
    data = data_tensor.cpu().detach().x.numpy()
    predicted_targets = model(data_tensor, data_tensor.edge_index).cpu()
else:
    data = data_tensor.detach().x.numpy()
    predicted_targets = model(data_tensor.x, data_tensor.edge_index)


print(torch.round((target_tensor - predicted_targets), decimals=1))
'''
color_map = colors.ListedColormap(['k','b','y','g','r'])

figure, axs = plot.subplots(2)
axs[0].scatter(data[:,0], data[:,1], c=data[:,n])#color_map[data[:,n]])
axs[1].scatter(data[:,0], data[:,1], c=predicted_classes.numpy())#color_map[predicted_classes])
plot.show()'''
