from IPython.display import clear_output
import matplotlib.pyplot as plt
import gym
import numpy

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from utils import *
from sac_v2_2Q import *

env = NormalizedActions(gym.make("Pendulum-v0"))
buffer_size = 1_000_000
gamma = 0.99
alpha = 0.2
soft_tau = 1e-2
q_lr = 3e-4
p_lr = 3e-4
alpha_lr = 3e-4

agent = SAC2019Agent(env, buffer_size, gamma, alpha, soft_tau, q_lr, p_lr, alpha_lr)

max_frames  = 40_000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 256

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > batch_size:
            agent.soft_q_update(batch_size)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            plot(frame_idx, rewards)
        
        if done:
            break
        
    rewards.append(episode_reward)