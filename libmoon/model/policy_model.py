import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, states_space, actions_space):
        super(Policy, self).__init__()
        self.states_space = states_space
        self.actions_space = actions_space

        self.base_model = torch.Sequential(
            nn.Linear(1, 60),
            nn.ReLU(),
            nn.Linear(60, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, state):
        pass
