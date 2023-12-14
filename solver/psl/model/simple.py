import torch
from torch import nn


class SimplePSLModel(nn.Module):

    def __init__(self, problem, args):

        '''
            :param dim: a 3d-array. [chanel, height, width]
            :param
        '''
        super().__init__()
        self.problem = problem
        self.args = args
        self.hidden_size = 128

        if 'lb' in dir(problem):
            self.psl_model = nn.Sequential(
                nn.Linear(self.problem.n_obj, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.args.n_var),
                nn.Sigmoid()
            )
        else:
            self.psl_model = nn.Sequential(
                nn.Linear(self.problem.n_obj, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.args.n_var),
            )

    def forward(self, pref):
        mid = self.psl_model(pref)
        if 'lb' in dir(self.problem):
            return mid * (self.problem.ub - self.problem.lb) + self.problem.lb
        else:
            return mid
