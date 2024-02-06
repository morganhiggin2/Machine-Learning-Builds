import numpy as np
import pandas
import os
import sys
import math
import matplotlib.pyplot as plt
import torch

#get data
data = pandas.read_csv('data/coin_flip_1.csv', index_col=False) 

#get flip 1 ratio
flip_ratio = data['flip'].value_counts(1)[0]

#create bernoulli distribution from data
plain_bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([flip_ratio]))

#create beta distrib off what we think the flip rate should be (0.5)

pred_beta = torch.distributions.beta.Beta(torch.tensor([5.0]), torch.tensor([5.0]))

print(torch.exp(plain_bern.log_prob(torch.tensor([1.0]))))

#the beta distribution can give us a sample for what it thinks the mean should be
print(pred_beta.rsample())

#lets get the mode of the beta distribution
print(pred_beta.mode)
