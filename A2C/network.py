import torch.nn as nn
import torch.nn.functional as F 

class Pol_Val_Network(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Pol_Val_Network, self).__init__()
        self.policy1 = nn.Linear(input_dim, 256)
        self.policy2 = nn.Linear(256, output_dim)

        self.value1 = nn.Linear(input_dim, 256)
        self.value2 = nn.Linear(256, 1)
    
    def forward(self, state):
        x = F.relu(self.policy1(state))
        x = self.policy2(x)

        value = F.relu(self.value1(state))
        value = self.value2(value)

        return x, value