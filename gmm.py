# Code to compare the alpha_t=(1-t), beta_t=t interpolant learning the GMM vs the dilated interpolant learning GMM
from learnGMM import run, gen
import matplotlib.pyplot as plt
import numpy as np

mode      = 'dVP'
N         = 4000 # dimensions
epochs    = 7000 # epochs
num_steps = 120  # time steps
bs_tr     = 256  # n_data
bs_te     = 1    # n_gen
device_id = 'cuda:0'

model = run(mode, N, epochs, num_steps, bs_tr, bs_te, device_id)