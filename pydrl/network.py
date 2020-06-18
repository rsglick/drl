import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau

class network(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size):
        super(network, self).__init__()
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.hidden_size  = hidden_size

        self.fc1 = nn.Linear(self.input_shape, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_shape)
    
    def forward(self, x):
        x = torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x 

    def get_action(self, x):
        action = self.forward(x)
        return action


import matplotlib.pyplot as plt

import gym
import numpy as np
import pandas as pd
import random

import time
import os

import pickle
import yaml

from onedof import onedof


torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False

# Env Seeds
#env = gym.make("MountainCarContinuous-v0")
#env = gym.make("Pendulum-v0")
env = onedof(state_start_pos=1.0)
#env = onedof()
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

input_shape  = np.prod(env.observation_space.shape)
output_shape = np.prod(env.action_space.shape)

hyperparams = {
"learning_rate"   : 1.0e-3,
"hidden_size"     : 8,
}


net = network(input_shape, output_shape, hyperparams["hidden_size"]).to(device)
net_optimizer = optim.Adam(list(net.parameters()), lr=hyperparams["learning_rate"])
#net_loss = torch.zeros(1).requires_grad_(True)
#net_scheduler = ReduceLROnPlateau(net_optimizer, verbose=True)

def net_train_test():
    obs = env.reset()

    action_ep_sum = 0
    num_ep     = 0
    timesteps  = 20000
    miss_dis   = []
    net_losses = []
    action_sums = []

    #net_loss = torch.zeros(1)

    trainFlag = True

    for i in range(timesteps):
         
        

        if trainFlag:
            action = net(obs.reshape(1,3))
        else:
            with torch.no_grad():
                action = net(obs.reshape(1,3))

        obs, rewards, dones, info = env.step(action)

        #env.render()

        k = 0.01
        #net_loss  = net_loss + k * action * action * env.dt

        #action_ep_sum += action.detach().numpy()
        action_ep_sum =  action_ep_sum + action

        if dones:
            obs = env.reset()

            kCtrl =    0.01 
            kTerm =  100.00
            net_loss = kTerm * info["delta_position"] * info["delta_position"] +\
                       kCtrl * action_ep_sum * action_ep_sum * env.dt
            #net_loss = torch.nn.SmoothL1Loss()

            num_ep += 1
            print(f'Ep({num_ep}):\t'
                  f'Loss={net_loss.detach().numpy().item():.2f},\t'
                  f'Miss={info["delta_position"]:.2f},\t'
                  f'ActSum={action_ep_sum.detach().numpy().item():.2f}')


            if trainFlag:
                net_optimizer.zero_grad()
                net_loss.backward()
                net_optimizer.step()

            action_sums.append(action_ep_sum.detach().numpy())
            miss_dis.append(info["delta_position"])
            net_losses.append(net_loss)
            #net_loss = torch.zeros(1)
            action_ep_sum = 0

    env.close()

    roll_window = 10
    action_sums = np.array(action_sums).flatten()
    roll_action = pd.DataFrame(action_sums).rolling(roll_window).mean() 
    miss_dis    = np.array(miss_dis)
    roll_miss   = pd.DataFrame(miss_dis).rolling(roll_window).mean() 
    net_losses  = np.array(net_losses)
    roll_losses = pd.DataFrame(net_losses).rolling(roll_window).mean()
    time = np.linspace(0, timesteps, timesteps)
    eps  = np.arange(0, num_ep)

    fig, ax = plt.subplots(1, sharex=True)
    #ax.plot( time, net_losses)
    ax.plot( eps, net_losses)
    ax.plot( eps, roll_losses)
    ax.set_title("net_losses")
    ax.set_ylabel("loss")
    ax.set_xlabel("Episode")

    fig, ax = plt.subplots(1, sharex=True)
    ax.plot( eps, miss_dis)
    ax.plot( eps, roll_miss)
    ax.set_title("miss_dis")
    ax.set_ylabel("miss_dis")
    ax.set_xlabel("Episode")

    fig, ax = plt.subplots(1, sharex=True)
    ax.plot( eps, action_sums)
    ax.plot( eps, roll_action)
    ax.set_title("action_sums")
    ax.set_ylabel("Action Sums")
    ax.set_xlabel("Episode")

