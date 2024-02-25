#seperate into 3 channels
#convolutional of k x k
#max pooling
#flatten amoung channels
#2 layer linear with relu on first, softmax into c classes on second

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy 

class TrafficSignClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = 

    def forward(self, X):
        return self.net(X)

