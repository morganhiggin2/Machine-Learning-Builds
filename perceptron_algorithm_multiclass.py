#to use this for multiple classes, simply run the algorithm to
#fit a line between each class
import torch
import torch.nn
import pandas
import matplotlib.pyplot as plot
import numpy

#read in data
data = pandas.read_csv('data/2d_target_data.csv', index_col=False)

data_tensor = torch.tensor(data.values, dtype=torch.float)
column_names = data.columns

class Perceptron2Classes(torch.nn.Module):
    def __init__(self, d):
        #pass our class model and our instance to the super class
        super().__init__()

        #store the degree of the polynomial
        self.d = d + 1
        
        #w = [a, b, c, d] (dx1)
        #x = [1, x, x^2, x^3] (dx1)
        #wm = [[a1, b1, c1, d1], [a2, b2, ...], ...]
        #xm = [
        #w_t * x = [1x1] (dxn)
        #for multiple x's, x becomes [[1, x1, ...], [1, x2, ...], ...]
        #w_t * x = (1xn)

        #set the parameters for a degree d polynomial
        self.params = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for i in range(self.d)]) 

    def parameters(self):
        return ([p for p in self.params])

    def forward(self, x):
        def basis_function(x_i, n):
            return torch.pow(x_i, n)

        param_tensor = torch.zeros((self.d, 1))

        for i in range(self.d):
            param_tensor[i, 0] = self.params[i]

        basis_function_tensor = torch.zeros((self.d, x.size(dim=1)))

        for i in range(self.d):
            basis_function_tensor[i, :] = basis_function(x, i)

        return param_tensor.transpose(0, 1).mm(basis_function_tensor)

def loss_fn(o, t):
    
    #if output * t > 0, return 0, else, return value
    return torch.sum(torch.mul(o, t).clamp(min=0)) 
        

model = Perceptron2Classes(1)
optim = torch.optim.SGD(
    [{'params': model.parameters()}],
    lr=1e-1,
)

for v in data_tensor:
    #optimizer zero gradient
    optim.zero_grad()

    #set output to the model output (the guess)
    output = model(v[0:1].unsqueeze(dim=0)).squeeze(dim=0)
    
    #get the loss of the new guess from the loss function
    loss = loss_fn(output, v[2:3].unsqueeze(dim=0))

    #backpropigate the loss function
    loss.backward()
    
    #update the optimizer with the new values
    optim.step()

colors = [0] * data.shape[0]

for i in range(len(data[column_names[1]])):
    colors[i] = 'orange' if data[column_names[1],i] == 1 else 'green'

plot.plot(data[column_names[0]], data[column_names[1]], c = colors)
plot.show()
