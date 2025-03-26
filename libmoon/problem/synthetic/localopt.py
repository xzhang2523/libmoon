from libmoon.problem.synthetic.mop import BaseMOP
'''
    This file solves the Pareto set learning with local optima issues.
    Problems are designed from easy to complex. 
'''
import torch
import numpy as np


class LFun1(BaseMOP):
    def __init__(self, n_var=30, n_obj=3, lbound=np.zeros(30),
                 ubound=np.ones(30)):
        '''
            n_obj can be set as any number. For simlicity, we set it as 3.
        '''
        lbound = np.zeros(n_var)
        ubound = np.ones(n_var)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound)

        self.problem_name = 'LFun1'

    def _evaluate_torch(self, x: torch.Tensor):
        '''
            This function is used to evaluate the objective values for the input x.
        '''
        f1 = torch.pow(x, 2) + 0.1 * torch.sin(5 * torch.pi * x)
        f2 = torch.pow((x-1), 2) + 0.1 * torch.cos(5 * torch.pi * x)
        return torch.stack([f1, f2], dim=1)


