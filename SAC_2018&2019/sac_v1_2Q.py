import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from utils import ReplayBuffer
from network import ValueNetwork, PolicyNetwork, SoftQNetwork


class SAC2018Agent(object):
    def __init__(self, env, buffer_size,
                 gamma, soft_tau, v_lr, q_lr, p_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.gamma = gamma
        self.soft_tau = soft_tau
        
        self.value_net = ValueNetwork(self.state_dim).to(self.device)
        self.target_value_net = ValueNetwork(self.state_dim).to(self.device)
        
        self.soft_q1_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.soft_q2_net = SoftQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.env.action_space).to(self.device)
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=v_lr)
        self.soft_q1_optimizer = optim.Adam(self.soft_q1_net.parameters(), lr=q_lr)
        self.soft_q2_optimizer = optim.Adam(self.soft_q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=p_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)

    def get_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate is False:
            action, _, _ = self.policy_net.sample(state)
        else:
            _, _, action = self.policy_net.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def soft_q_update(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
    
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        expected_q1_value = self.soft_q1_net(state, action)
        expected_q2_value = self.soft_q2_net(state, action)

        expected_value   = self.value_net(state)
        new_action, log_prob, _ = self.policy_net.sample(state)


        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value
        q1_value_loss = F.mse_loss(expected_q1_value, next_q_value.detach())
        q2_value_loss = F.mse_loss(expected_q2_value, next_q_value.detach())

        expected_new_q_value = torch.min(self.soft_q1_net(state, new_action), self.soft_q2_net(state, new_action))
        next_value = expected_new_q_value - log_prob
        value_loss = F.mse_loss(expected_value, next_value.detach())
        
        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        
        self.soft_q1_optimizer.zero_grad()
        q1_value_loss.backward()
        self.soft_q1_optimizer.step()

        self.soft_q2_optimizer.zero_grad()
        q2_value_loss.backward()
        self.soft_q2_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
    
        return q1_value_loss.item(), q2_value_loss.item(), value_loss.item(), policy_loss.item()
