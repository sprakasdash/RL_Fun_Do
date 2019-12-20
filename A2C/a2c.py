import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from network import Pol_Val_Network

class A2C_Agent():
    def __init__(self, env, gamma, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.gamma = gamma
        self.lr = lr

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.model = Pol_Val_Network(self.obs_dim, self.act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        x, _ = self.model.forward(state)
        dist = F.softmax(x, dim=0)
        probs = Categorical(dist)
        return probs.sample().cpu().detach().item()

    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sarn[0] for sarn in trajectory]).to(self.device)
        actions = torch.FloatTensor([sarn[1] for sarn in trajectory]).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor([sarn[2] for sarn in trajectory]).to(self.device)
        next_state = torch.FloatTensor([sarn[3] for sarn in trajectory]).to(self.device)
        dones = torch.FloatTensor([sarn[0] for sarn in trajectory]).view(-1, 1).to(self.device)
        #SARN - state, action, reward, next_state
        # calculate discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in \
            range(reward[j:].size(0))]) * reward[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        x, values = self.model.forward(states)
        dists = F.softmax(x, dim=1)
        probs = Categorical(dists)
        advantage = value_targets - values

        #calculate value loss
        value_loss = F.mse_loss(values, value_targets.detach())

        #calculate entropy
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torc.stack(entropy).sum()

        #calculate policy loss
        policy_loss = - probs.log_prob(action.view(action.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        total_loss = policy_loss + value_loss - 0.001 * entropy
        return total_loss

    def update(self, trajectory):
        loss = self.compute_loss(trajectory)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()