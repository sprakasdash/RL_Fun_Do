from IPython.display import clear_output
import matplotlib.pyplot as plt
import gym
import numpy

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from datetime import datetime
import itertools
from tensorboardX import SummaryWriter

from utils import *
from sac_v2_2Q import *

# ---IF YOU ARE USING pybullet ENVIRONMENT------
# pybullet.connect(pybullet.DIRECT)
# env = gym.make("AntBulletEnv-v0")
# ----------------------------------------------
env = gym.make("Ant-v2")
buffer_size = 1_000_000
max_time_steps = 1_000_001 #which we generally use in research papaers
start_steps = 10000
batch_size  = 256
gamma = 0.99
alpha = 0.2
soft_tau = 1e-2
q_lr = 3e-4
p_lr = 3e-4
alpha_lr = 3e-4

agent = SAC2019Agent(env, buffer_size, gamma, alpha, soft_tau, q_lr, p_lr, alpha_lr)

total_numsteps = 0

writer = SummaryWriter(logdir='runs/{}_SAC_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), env.unwrapped.spec.id))

# STORING IN THE /runs directory in you current pwd...
# Then you could call:
# tensorboard --logdir /runs to view the results

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    
    while not done:
        if start_steps > total_numsteps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)
        
        if len(agent.replay_buffer) > batch_size:
            soft_q1_loss, soft_q2_loss, policy_loss, entropy_loss, alpha = agent.soft_q_update(batch_size)
            writer.add_scalar('loss/soft_q1', soft_q1_loss)
            writer.add_scalar('loss/soft_q2', soft_q2_loss)
            writer.add_scalar('loss/policy', policy_loss)
            writer.add_scalar('loss/entropy_loss', entropy_loss)
            writer.add_scalar('temperature/alpha', alpha)
        
        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        if episode_steps == env._max_episode_steps:
            done = 1
        else:
            float(not done)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
    
    if total_numsteps > max_time_steps:
        break
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, Total Steps: {}, Episode Steps: {}, Reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
    if i_episode % 10 == 0:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes
        
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        print("Test Episode: {}, Average Reward: {}".format(episodes, round(avg_reward, 2)))
env.close()
