import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import gym
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space

import collections
import numpy as np

import time
import random
import os

from onedof import onedof

def preprocess_obs_space(obs_space: Space, device: str):
    """
    The `preprocess_obs_fn` receives the observation `x` in the shape of
    `(batch_num,) + obs_space.shape`.
    1) If the `obs_space` is `Discrete`, `preprocess_obs_fn` outputs a
    preprocessed obs in the shape of
    `(batch_num, obs_space.n)`.
    2) If the `obs_space` is `Box`, `preprocess_obs_fn` outputs a
    preprocessed obs in the shape of
    `(batch_num,) + obs_space.shape`.
    In addition, the preprocessed obs will be sent to `device` (either
    `cpu` or `cuda`)
    """
    if isinstance(obs_space, Discrete):
        def preprocess_obs_fn(x):
            return F.one_hot(torch.LongTensor(x), obs_space.n).float().to(device)
        return (obs_space.n, preprocess_obs_fn)

    elif isinstance(obs_space, Box):
        def preprocess_obs_fn(x):
            return torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1).to(device)
        return (np.array(obs_space.shape).prod(), preprocess_obs_fn)

    else:
        raise NotImplementedError("Error: the model does not support input space of type {}".format(
            type(obs_space).__name__))

def preprocess_ac_space(ac_space: Space):
    if isinstance(ac_space, Discrete):
        return ac_space.n

    elif isinstance(ac_space, MultiDiscrete):
        return ac_space.nvec.sum()

    elif isinstance(ac_space, Box):
        return np.prod(ac_space.shape)

    else:
        raise NotImplementedError("Error: the model does not support output space of type {}".format(
            type(ac_space).__name__))


torch.autograd.set_detect_anomaly(True)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False

# Env Seeds
#env = gym.make("MountainCarContinuous-v0")
env = gym.make("Pendulum-v0")
#env = onedof()
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

#input_shape  = np.prod(env.observation_space.shape)
#output_shape = np.prod(env.action_space.shape)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)



##############################################################################

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

hidden_size=64
# ALGO LOGIC: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_shape)
        self.logstd = nn.Linear(hidden_size, output_shape)
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (env.action_space.high - env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (env.action_space.high + env.action_space.low) / 2.)
    
    def forward(self, x):
        x = preprocess_obs_fn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.logstd(x)
        log_std = torch.clamp(log_std, min=-20., max=2)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)


class SoftQNetwork(nn.Module):
    def __init__(self):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape+output_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, a):
        x = preprocess_obs_fn(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##############################################################################
#
# Hyperparams
# 
hyperparams = {
"buffer_size"     : int(1e6),
"batch_size"      :  64,
"learning_rate"   : 3.0e-4,
"learning_starts" : 1000,
"tau"             : 0.005,
"gamma"           : 0.99,
}

total_timesteps = 10000
try:
    episode_length  = int(env.max_episode_steps)
except:
    episode_length  = env.spec.max_episode_steps



writer = SummaryWriter(f"testing")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in hyperparams.items()])))

# Replay Buffer
rb = ReplayBuffer(hyperparams["buffer_size"])

# Actor Network
pg = Policy().to(device)

# Critic Networks
qf1 = SoftQNetwork().to(device)
qf2 = SoftQNetwork().to(device)
qf1_target = SoftQNetwork().to(device)
qf2_target = SoftQNetwork().to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

# Entropy
entropy_coef = 1.0 
init_alpha   = 1.0
target_alpha = - np.prod(env.action_space.shape).astype(np.float32)
log_alpha    = torch.log(torch.ones(1, device=device) * init_alpha).requires_grad_(True)

values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=hyperparams["learning_rate"])
policy_optimizer = optim.Adam(list(pg.parameters()), lr=hyperparams["learning_rate"])
alpha_optimizer  = optim.Adam([log_alpha], lr=hyperparams["learning_rate"])

loss_fn = nn.MSELoss()


global_step = 0
while global_step < total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((episode_length,), dtype=object)
    rewards, dones = np.zeros((2, episode_length))
    qf1_losses, qf2_losses, policy_losses = np.zeros((3, episode_length))
    obs = np.empty((episode_length,) + env.observation_space.shape)
    
    # ALGO LOGIC: put other storage logic here
    entropys = torch.zeros((episode_length,), device=device)
    alpha_losses = []
    alphas = []
    
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(episode_length):
        global_step += 1
        obs[step] = next_obs.copy()
        
        # ALGO LOGIC: put action logic here
        action, log_prop, _ = pg.get_action(obs[step:step+1])
        actions[step] = action.tolist()[0]

        # Alpha Entropy Optimize
        alpha_loss = None
        entropy_coef = torch.exp( log_alpha.detach())
        alpha_loss = - ( log_alpha * ( log_prop + target_alpha).detach()).mean()
        
        alpha_losses.append(alpha_loss.item())
        alphas.append(entropy_coef.item())

        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha = entropy_coef

    
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(action.tolist()[0])
        rb.put((obs[step], actions[step], rewards[step], next_obs, dones[step]))
        next_obs = np.array(next_obs)
        # ALGO LOGIC: training.
        if len(rb.buffer) > hyperparams["learning_starts"]:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(hyperparams["batch_size"])
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = pg.get_action(s_next_obses)
                qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions)
                qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * hyperparams["gamma"] * (min_qf_next_target).view(-1)

            qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device)).view(-1)
            qf2_a_values = qf2.forward(s_obs, torch.Tensor(s_actions).to(device)).view(-1)
            # qf1, qf2 = critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = loss_fn(qf1_a_values, next_q_value)
            qf2_loss = loss_fn(qf2_a_values, next_q_value)

            pi, log_pi, _ = pg.get_action(s_obs)
            qf1_pi = qf1.forward(s_obs, pi)
            qf2_pi = qf2.forward(s_obs, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)

            policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

            values_optimizer.zero_grad()
            qf1_loss.backward()
            values_optimizer.step()

            values_optimizer.zero_grad()
            qf2_loss.backward()
            values_optimizer.step()
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # update the target network
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(hyperparams["tau"] * param.data + (1 - hyperparams["tau"]) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(hyperparams["tau"]* param.data + (1 - hyperparams["tau"]) * target_param.data)
            qf1_losses[step] = qf1_loss.item()
            qf2_losses[step] = qf2_loss.item()
            policy_losses[step] = policy_loss.item()

        if dones[step]:
            break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    print(f"global_step={global_step}, episode_reward={rewards.sum()}")
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("losses/entropy", entropys[:step+1].mean().item(), global_step)
    writer.add_scalar("losses/soft_q_value_1_loss", qf1_losses[:step+1].mean(), global_step)
    writer.add_scalar("losses/soft_q_value_2_loss", qf2_losses[:step+1].mean(), global_step)
    writer.add_scalar("losses/policy_loss", policy_losses[:step+1].mean(), global_step)
writer.close()
env.close()

