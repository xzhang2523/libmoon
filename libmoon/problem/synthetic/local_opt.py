from libmoon.problem.synthetic.mop import BaseMOP
'''
    This file solves the Pareto set learning with local optima issues. 
'''
import torch
import numpy as np


class Fun1(BaseMOP):
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
        self.problem_name = 'Fun1'

    def _evaluate_torch(self, x: torch.Tensor):
        '''
            This function is used to evaluate the objective values for the input x.
        '''
        pass


