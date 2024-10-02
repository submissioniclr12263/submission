import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run(mode, N, epochs, num_steps, bs_tr, bs_te, device_id):
    device = torch.device(device_id)
    
    class NN(nn.Module):
        def __init__(self):
            super(NN, self).__init__()
            ones = torch.ones(N, device=device)
            self.c = torch.nn.Parameter(torch.randn(num_steps))#b(ts)/d(ts)
            self.u = torch.nn.Parameter(torch.randn(N, num_steps))#ones * a(ts)**2/d(ts)
            self.w = torch.nn.Parameter(torch.randn(N, num_steps))#ones * b(ts)*N/d(ts)
            self.b = torch.nn.Parameter(torch.randn(num_steps))#.693
        
        def forward(self, x):
            tanh = torch.tanh(self.b + torch.einsum('ij,ikj->kj', self.w, x) / (np.sqrt(N) * c))
            return self.c * x + torch.einsum('ij,kj->ikj', self.u, tanh)
    
    criterion = nn.MSELoss()
    
    p  = torch.tensor(.8)
    h  = -torch.log((1 / p) - 1)/2
    s1 = torch.sign(2.*(torch.rand(bs_tr, device=device) < p) - 1)
    m1 = torch.randn(N, bs_tr, device=device) + torch.outer(torch.ones(N, device=device), s1)
    
    if mode == 'VP':
        c, end = 1, 1
        a  = lambda t: 1-t
        ap = lambda t: -1
        b  = lambda t: t
        bp = lambda t: 1
    elif mode == 'dVP':
        c, end, C = 1, 2, 4
        a  = lambda t: (1 - C/np.sqrt(N) * t) * (t < 1) + (2-t) * (1-C/np.sqrt(N)) * (t >= 1)
        ap = lambda t: - C/np.sqrt(N) * (t < 1)         + -(1-C/np.sqrt(N)) * (t >= 1)
        b  = lambda t: C/np.sqrt(N) * t * (t < 1)       + (C/np.sqrt(N) + (1-C/np.sqrt(N)) * (t - 1)) * (t >= 1)
        bp = lambda t: C/np.sqrt(N) * (t < 1)           + (1-C/np.sqrt(N)) * (t >= 1)
        
    d  = lambda t: a(t)**2 + b(t)**2
    ts = torch.linspace(0, end, num_steps+1, device=device)[:-1]
    losses = []
    
    model = NN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for i in range(epochs):
        z = torch.randn((N, bs_tr, num_steps), device=device)
        mt = a(ts) * z + b(ts) * m1[:, :, None]
        den_pred = model(mt)
        loss = criterion(den_pred, m1[:, :, None])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        if i % 100 == 0: print(i) 
    
    return model

def gen(mode, N, num_steps, bs_te, device_id, model):
    device = torch.device(device_id)
    
    if mode == 'VP':
        c, end = 1, 1
        a  = lambda t: 1-t
        ap = lambda t: -1
        b  = lambda t: t
        bp = lambda t: 1
    elif mode == 'dVP':
        c, end, C = 1, 2, 3
        a  = lambda t: (1 - C/np.sqrt(N) * t) * (t < 1) + (2-t) * (1-C/np.sqrt(N)) * (t >= 1)
        ap = lambda t: - C/np.sqrt(N) * (t < 1)         + -(1-C/np.sqrt(N)) * (t >= 1)
        b  = lambda t: C/np.sqrt(N) * t * (t < 1)       + (C/np.sqrt(N) + (1-C/np.sqrt(N)) * (t - 1)) * (t >= 1)
        bp = lambda t: C/np.sqrt(N) * (t < 1)           + (1-C/np.sqrt(N)) * (t >= 1)

    ts = torch.linspace(0, end, num_steps+1, device=device)[:-1]
    mgen = torch.randn((N, bs_te, num_steps), device=device) * c

    for i in range(num_steps-1):
        t = ts[i]
        if i % 20 == 0: print(t)
        with torch.no_grad():
            bf = ap(t) / a(t) * mgen[:, :, i] + (bp(t) - ap(t)/a(t)*b(t)) * model(mgen)[:, :, i]
            mgen[:, :, i+1] = mgen[:, :, i] + bf * (ts[1] - ts[0])

    return mgen.cpu().numpy()
