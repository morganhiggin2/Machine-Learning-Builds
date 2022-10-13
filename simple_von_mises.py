import torch
import numpy as np
import pandas
import sys
import scipy
#get the data
data = pandas.read_csv('data/von_mises_1', index_col=False)

#transform data to span from 0 to 2pi
spread = data['value'].max() - data['value'].min()
lower = data['value'].min()
data['value'] = (data['value'] - lower) / spread * 2 * np.pi

#get the mls for theta
theta_0 = np.arctan(np.sin(data['value']).sum() / np.cos(data['value']).sum())

#get A(kappa)
n = data.shape[0]
besel_complex = ((1 / n) * np.cos(data['value']).sum()) * np.cos(theta_0) + ((1 / n) * np.sin(data['value']).sum()) * np.cos(theta_0)

#use optimization to get find m that satisfies the equation
def kappa_equation(kappa):
    (torch.special.i1(kappa) / torch.special.i0(kappa)) - torch.tensor(besel_complex)

#set initial kappa guess
kappa = torch.tensor(np.pi)

#optimize so kappa_equation is 0 for the given value of kappa


#other option: use scikit learn to generate values for distribution
kappa, loc, scale = scipy.stats.vonmises.fit(data['value'].to_numpy(), fscale=1)

von_misses_distr = torch.distributions.von_mises.VonMises(torch.tensor([loc]), torch.tensor([kappa]))
