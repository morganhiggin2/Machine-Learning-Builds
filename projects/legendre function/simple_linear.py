import torch
import torch.nn

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()

        linear_net = torch.nn.Linear(3, 1, dtype=torch.float32)
    
        self.net = torch.nn.Sequential(linear_net)

    def forward(self, X):
        pred = self.net(X)
        return pred


optim = torch.optim.SGD([{'params': model.parameters()}], lr = 1e-2)
loss_fn = torch.nn.MSELoss()
data_tensor = requires_grad_(True)

for epoch in range(50):
    output = model(data)

    loss = loss_fn(output, data)

    optim.zero_grad()

    loss.backward()

    optim.step()

    if epock % 10 == 0:
        print(str(epoch) + " " + str(loss))
