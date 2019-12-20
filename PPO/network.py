import gym 
import torch
import torch.nn as nn
import torch.nn.functional as F 


seed = 1
env = gym.make('Pendulum-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
torch.manual_seed(seed)
env.seed(seed)
gamma = 0.9

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.mu_head = nn.Linear(8, 1)
        self.sigma_head = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return mu, sigma
        