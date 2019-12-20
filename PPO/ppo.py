import pickle
import os
import gym
import torch 
from torch.nn import utils
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])

seed = 1
env = gym.make('Pendulum-v0')
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

class PPO_Agent():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 32
    
    def __init__(self):
        super(PPO_Agent, self).__init__()
        self.critic_net = Critic().float()
        self.actor_net = Actor().float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.critic_optimizer = Adam(self.critic_net.parameters(), 4e-4)
        self.actor_optimizer = Adam(self.actor_net.parameters(), 1e-4)
        
        # if not os.path.exists('../param'):
        #     os.makedirs('../param/net_param')
        #     os.makedirs('../param/img')
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.actor_net(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2.0, 2.0)
        return action.item(), action_log_prob.item()
    
    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    
    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'param/ppo_actor_net.pkl')
        torch.save(self.critic_net.state_dict(), 'param/ppo_critic_net.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return (self.counter % self.buffer_capacity == 0)

    def update(self):
        self.training_step += 1

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)
        
        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            target_v = reward + gamma *self.critic_net(next_state)
        advantage = (target_v - self.critic_net(state)).detach()
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                (mu, sigma) = self.actor_net(state[index])
                dist = Normal(mu, sigma)
                action_log_prob = dist.log_prob(action[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                loss1 = ratio * advantage[index]
                loss2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage[index]
                action_loss = - torch.min(loss1, loss2).mean() #MAX-MIN DESCENT

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
        del self.buffer[:]

def main():
    env = gym.make('Pendulum-v0')
    env.seed = seed

    agent = PPO_Agent()
    training_records = []
    running_reward = -1000

    for i in range(1000):
        score = 0
        state = env.reset()
        env.render()
        for t in range(200):
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step([action])
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
            with open('log/ppo_training_records.pkl', 'wb') as f:
                pickle.dump(training_records, f)
            break
    plt.plot([r_ep for r in training_records], [r_rew for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving avg eps rewards')
    plt.savefig("img/ppo.jpg")
    plt.show()


if __name__ == '__main__':
    main()