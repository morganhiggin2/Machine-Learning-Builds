# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:25:56 2022

@author: morga
"""

import pandas as pd
import torch

#import data though pandas
datatable = pd.read_csv("data/simple_1.csv")

#convert columns into tensors
samples = torch.tensor(datatable["sample"].values)
targets = torch.tensor(datatable["target"].values)

together = torch.stack([samples, targets])

print(together)

print(torch.cov(together))