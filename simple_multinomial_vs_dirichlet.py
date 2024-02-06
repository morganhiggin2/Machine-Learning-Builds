import pandas
import numpy as np
import torch
import os
import sys

#get the data
data = pandas.read_csv('data/multinomial_data_1.csv', index_col=False)

#sums
sums = data.sum() / data.shape[0]

#make distribution for data
mul_dis = torch.distributions.multinomial.Multinomial(1, torch.tensor([sums['one'], sums['two'], sums['three']]))

#get sample
print(mul_dis.sample())

senario = [0, 0, 1]

#get probability for senario
print("probability for senario " + ''.join(str(x) for x in senario) + " is ")

print(torch.exp(mul_dis.log_prob(torch.tensor(senario))))

#make dirichlet distribution concentration based off initial guess
guess_conc = [0.1, 0.1, 0.8]

#make dirichlet distribution
dir_dist = torch.distributions.dirichlet.Dirichlet(torch.tensor(guess_conc))

#get sample
print(dir_dist.sample())
print(dir_dist.rsample())
print(dir_dist.sample())
