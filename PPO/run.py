import os
import torch
import numpy as np 
from collections import namedtuple
import gym 
import matplotlib.pyplot as plt 

from network import Actor, Critic
from ppo import *

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])

agent = PPO_Agent()
training_records = []
running_reward = -1000

for i in range(1000):
    score = 0
    state = env.reset()
    env.render()
    for t in range(200):
        action, action_log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        trans = Transition(state, action, reward, action_log_prob, next_state)
        env.render()
        if agent.store_transition(trans):
            agent.update()
        score += reward
        state = next_state

    running_reward = running_reward * 0.9 + score * 0.1
    training_records.append(TrainRecord(i, running_reward))
    if i % 10 == 0:
        print("Epoch {}, Moving average score is: {:.2f} ".format(i, running_reward))
    if running_reward > -200:
        print("Solved! Moving average score is now {}!".format(running_reward))
        env.close()
        agent.save_param()
        break

