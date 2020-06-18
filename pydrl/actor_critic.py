import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

class actor(nn.Module):
    def __init__(self, env, input_shape, output_shape, hidden_size,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(actor, self).__init__()
        self.env = env
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.hidden_size  = hidden_size
        self.device       = device

        self.fc1 = nn.Linear(self.input_shape, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mean = nn.Linear(self.hidden_size, self.output_shape)
        self.logstd = nn.Linear(self.hidden_size, self.output_shape)
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (self.env.action_space.high - self.env.action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            (self.env.action_space.high + self.env.action_space.low) / 2.)
    
    def forward(self, x):
        x = torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std_tmp = self.logstd(x)
        log_std = torch.clamp(log_std_tmp, min=-20., max=2)
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
        return super(actor, self).to(device)


class critic(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(critic, self).__init__()
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.hidden_size  = hidden_size
        self.device       = device

        self.first_layer_size = self.input_shape + self.output_shape
        self.fc1 = nn.Linear(self.first_layer_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)

    def forward(self, x, a):
        x = torch.Tensor(x).float().view(torch.Tensor(x).shape[0], -1).to(self.device)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

