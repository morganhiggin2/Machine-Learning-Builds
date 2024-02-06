import torch
import torch.nn
import pandas
import matplotlib.pyplot as plot
import numpy

#read in data
data = pandas.read_csv('data/polynomial_1.csv', index_col=False)

data_tensor = torch.tensor(data.values, dtype=torch.float)
column_names = data.columns

class PolynomialD(torch.nn.Module):
    def __init__(self, d):
        #pass our class model and our instance to the super class
        super().__init__()

        #store the degree of the polynomial
        self.d = d + 1

        #set the parameters for a degree d polynomial
        self.params = [torch.nn.Parameter(torch.randn(1)) for i in range(self.d)] 

    def parameters(self):
        return ([p for p in self.params])

    def forward(self, x):
        
        total = torch.tensor([0])

        for i in range(self.d):
            total = total + self.params[i] * x**i

        return total
        '''
        #get the value from the x tensor
        x_value = x[0].item()

        #create lambda function for powering x by the index
        f = lambda i : x**i

        #create the x value vector
      
        xs = torch.tensor([f(i) for i in range(self.d)], dtype=torch.float)
        print(torch.dot(torch.transpose(xs, 0, 0), torch.tensor(self.params, dtype=torch.float)))
        #return the value of the function evaluated at the point x with the parameters params
        return torch.dot(torch.transpose(xs, 0, 0), torch.tensor(self.params, dtype=torch.float)).unsqueeze(dim=0)
        '''

model = PolynomialD(2)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(
    [{'params': model.parameters()}],
    lr=1e-1,
    weight_decay=0.0
)

for i, t in data_tensor:
    #optimizer zero gradient
    optim.zero_grad()

    #set output to the model output (the guess)
    output = model(i.unsqueeze(dim=0))
    
    #get the loss of the new guess from the loss function
    loss = loss_fn(output, t.unsqueeze(dim=0))
    
    #backpropigate the loss function
    loss.backward()
    
    #update the optimizer with the new values
    optim.step()

predicted_values = model(torch.reshape(data_tensor[0:,0], (data_tensor.size(dim=0),1))).detach().numpy()
plot.plot(data[column_names[0]], data[column_names[1]])
plot.plot(data[column_names[0]], predicted_values)
plot.show()
