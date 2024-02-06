import torch
import torch.nn
import pandas
import matplotlib.pyplot as plot

#read in data
data = pandas.read_csv('data/log_fit_1.csv', index_col=False)

data_tensor = torch.tensor(data.values, dtype=torch.float)

class LeastSquaresModel(torch.nn.Module):
    def __init__(self):
        #pass our class model and our instance to the super class
        #and call the initializer
        super(LeastSquaresModel, self).__init__()

        #one input and one output
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        #get the guess from the linear model
        prediction = self.linear(x)

        #return prediction
        return prediction

model = LeastSquaresModel()
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(
    [{'params': model.parameters()}],
    lr=1e-2,
    weight_decay=0.0
)

for i, t in data_tensor:
    #optimizer zero gradient
    optim.zero_grad()

    #set output to the model output (the guess)
    output = model(i.unsqueeze(dim=0))

    #get the loss of the new guess from the loss function
    loss = loss_fn(output, t.unsqueeze(dim=0))
    print(loss)
    #backpropigate the loss function
    loss.backward()
    
    #update the optimizer with the new values
    optim.step()

predicted_values = model(torch.reshape(data_tensor[0:,0], (data_tensor.size(dim=0),1))).detach().numpy()
plot.plot(data['timestamp'], data['target'])
plot.plot(data['timestamp'], predicted_values)
plot.show()
