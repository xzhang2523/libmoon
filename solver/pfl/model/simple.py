
import torch
from torch import nn


class PFLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = 128
        self.model = nn.Sequential(
            nn.Linear(self.args.n_obj, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.args.n_obj)
        )

    def forward(self, x):
        '''
        :param x: input
        :return: the predicted Pareto front
        '''
        raise NotImplementedError

    def get_pf(self):
        '''
        :return: the true Pareto front
        '''
        raise NotImplementedError
