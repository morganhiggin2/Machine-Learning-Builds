import torch
import torch.nn
import pandas
import matplotlib.pyplot as plot
import numpy

#read in data
data = pandas.read_csv('data/log_fit_1.csv', index_col=False)

data_tensor = torch.tensor(data.values, dtype=torch.float)

column_names = data.columns

class KClassLinear(torch.nn.Module):
    def __init__(self, k):
        #pass our class model and our instance to the super class
        super().__init__()
        
        #number of classes
        self.k = k

        #set the parameters for a degree d polynomial
        self.params = [torch.nn.Parameter(torch.randn(1)), torch.nn.Parameter(torch.randn(1))] 

    def parameters(self):
        return ([p for p in self.params])

    def forward(self, x):
        #loop though k functions
        #pick function with highest value
        
model = Log()
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(
    [{'params': model.parameters()}],
    lr=1e-1,
    weight_decay=0.2
)

x_values = data_tensor[0:,0]
y_values = data_tensor[0:,1]
final_loss = -1

for epoch in range(500):
    #optimizer zero gradient
    optim.zero_grad()

    #set output to the model output (the guess)
    output = model(x_values)
    
    #get the loss of the new guess from the loss function
    loss = loss_fn(output, y_values)
    
    #backpropigate the loss function
    loss.backward()
    
    #update the optimizer with the new values
    optim.step()

    final_loss = loss

print(final_loss)

predicted_values = model(torch.reshape(data_tensor[0:,0], (data_tensor.size(dim=0),1))).detach().numpy()
plot.plot(data[column_names[0]], data[column_names[1]])
plot.plot(data[column_names[0]], predicted_values)
plot.show()
