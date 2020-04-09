import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from utils import ReplayBuffer
from network import PolicyNetwork, SoftQNetwork

class SAC2019Agent(object):
    def __init__ (self, env, buffer_size,
                  gamma, alpha, soft_tau, q_lr, p_lr, alpha_lr):
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.gamma = gamma
        self.soft_tau = soft_tau
        
        self.soft_q1_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.soft_q2_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_soft_q1_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_soft_q2_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        
        for target_param, param in zip(self.target_soft_q1_net.parameters(), self.soft_q1_net.parameters()):
            target_param.data.copy_(param)
        
        for target_param, param in zip(self.target_soft_q2_net.parameters(), self.soft_q2_net.parameters()):
            target_param.data.copy_(param)
        
        self.soft_q1_optim = optim.Adam(self.soft_q1_net.parameters(), lr=q_lr)
        self.soft_q2_optim = optim.Adam(self.soft_q2_net.parameters(), lr=q_lr)
        self.policy_optim =  optim.Adam(self.policy_net.parameters(), lr=p_lr)
        
        self.alpha = alpha
        self.target_entropy = - T.prod(T.Tensor(self.env.action_space.shape).to(self.device)).item()
#         self.target_entropy = self.env.action_space.shape[0]
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)
        
    def get_action(self, state):
        state = T.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.rsample()
        action = T.tanh(z)
        action = action.cpu().detach().numpy()[0]
        return action
    
    def soft_q_update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = T.FloatTensor(states).to(self.device)
        actions = T.FloatTensor(actions).to(self.device)
        rewards = T.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = T.FloatTensor(next_states).to(self.device)
        dones = T.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)
        #----------------------reward scaling-------------------------
#         reward = reward_scale * (reward - reward.mean(dim=0)) / ((reward.std(dim=0)) + 1e-6)
        
        next_actions, log_prob, z, mean, log_std = self.policy_net.evaluate(states)
        new_next_actions, new_log_prob, *_ = self.policy_net.evaluate(next_states)
        
#         alpha_loss = (self.log_alpha * (- log_prob - self.target_entropy).detach()).mean()
#         self.alpha_optim.zero_grad()
#         alpha_loss.backward()
#         self.alpha_optim.step()
#         self.alpha = self.log_alpha.exp()
        
        curr_q1_val = self.soft_q1_net.forward(states, actions)
        curr_q2_val = self.soft_q2_net.forward(states, actions)
        min_q = T.min(self.soft_q1_net.forward(states, next_actions), self.soft_q2_net.forward(states, next_actions))
        next_q_val = T.min(self.soft_q1_net.forward(next_states, new_next_actions), self.soft_q2_net.forward(next_states, new_next_actions)) - self.alpha * new_log_prob
        next_q_val = rewards + (1 - dones) * self.gamma * next_q_val
        
        soft_q1_loss = F.mse_loss(curr_q1_val, next_q_val.detach())
        soft_q2_loss = F.mse_loss(curr_q2_val, next_q_val.detach())
        
        self.soft_q1_optim.zero_grad()
        soft_q1_loss.backward()
        self.soft_q1_optim.step()
        
        self.soft_q2_optim.zero_grad()
        soft_q2_loss.backward()
        self.soft_q2_optim.step()
        
        policy_loss = (self.alpha * log_prob - min_q).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        alpha_loss = (self.log_alpha * (- log_prob - self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
        for target_param, param in zip(self.target_soft_q1_net.parameters(), self.soft_q1_net.parameters()):
            target_param.data.copy_(
            self.soft_tau * param + (1 - self.soft_tau) * target_param
            )
        for target_param, param in zip(self.target_soft_q1_net.parameters(), self.soft_q1_net.parameters()):
            target_param.data.copy_(
            self.soft_tau * param + (1 - self.soft_tau) * target_param
            )