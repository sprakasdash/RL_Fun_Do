import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from utils import ReplayBuffer
from network import PolicyNetwork, SoftQNetwork

class SAC2019Agent(object):
    def __init__ (self, env, buffer_size,
                  gamma, alpha, soft_tau, q_lr, p_lr, alpha_lr):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.gamma = gamma
        self.alpha = alpha
        self.soft_tau = soft_tau
        
        self.soft_q1_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.soft_q2_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_soft_q1_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_soft_q2_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.env.action_space).to(self.device)
        
        for target_param, param in zip(self.target_soft_q1_net.parameters(), self.soft_q1_net.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_soft_q2_net.parameters(), self.soft_q2_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.soft_q1_optim = optim.Adam(self.soft_q1_net.parameters(), lr=q_lr)
        self.soft_q2_optim = optim.Adam(self.soft_q2_net.parameters(), lr=q_lr)
        self.policy_optim =  optim.Adam(self.policy_net.parameters(), lr=p_lr)
        
        self.target_entropy = - torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
#         self.target_entropy = self.env.action_space.shape[0]
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)
        
    def get_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate is False:
            action, _, _ = self.policy_net.sample(state)
        else:
            _, _, action = self.policy_net.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def soft_q_update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)
        #----------------------reward scaling-------------------------
#         reward = reward_scale * (reward - reward.mean(dim=0)) / ((reward.std(dim=0)) + 1e-6)
        
        
        new_next_actions, new_log_prob, _ = self.policy_net.sample(next_states)
        next_q_val = torch.min(self.soft_q1_net.forward(next_states, new_next_actions), self.soft_q2_net.forward(next_states, new_next_actions)).to(self.device) - self.alpha * new_log_prob
        next_q_val = (rewards + (1 - dones) * self.gamma * next_q_val).to(self.device)
            
        next_actions, log_prob, _ = self.policy_net.sample(states)
        
        curr_q1_val = self.soft_q1_net.forward(states, actions)
        curr_q2_val = self.soft_q2_net.forward(states, actions)
        min_q = torch.min(self.soft_q1_net.forward(states, next_actions), self.soft_q2_net.forward(states, next_actions)).to(self.device)
        
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
        alpha_logs = self.alpha.clone()
        
        for target_param, param in zip(self.target_soft_q1_net.parameters(), self.soft_q1_net.parameters()):
            target_param.data.copy_(
            self.soft_tau * param.data + (1 - self.soft_tau) * target_param.data
            )
        for target_param, param in zip(self.target_soft_q1_net.parameters(), self.soft_q1_net.parameters()):
            target_param.data.copy_(
            self.soft_tau * param.data + (1 - self.soft_tau) * target_param.data
            )
        return soft_q1_loss.item(), soft_q2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_logs.item()
