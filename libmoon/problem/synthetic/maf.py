from numpy import array
import torch
import numpy as np
from libmoon.problem.synthetic.mop import BaseMOP
'''
    Reference: 
'''

class MAF1(BaseMOP):
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
        self.problem_name = 'MAF1'

    def _evaluate_torch(self, x: torch.Tensor):
        if type(x) == torch.Tensor:
            g = torch.sum( torch.pow(x[:, 2:] - 0.5, 2), dim=1 )
            f1 = (1 - x[:,0] * x[:,1]) * (1 + g)
            f2 = (1 - x[:,0] * (1 - x[:,1]) ) * (1 + g)
            f3 = x[:,0] * (1 + g)
            return torch.stack((f1, f2, f3), dim=1)/2
        else:
            assert False

    def get_pf(self):
        return array([[0.0, 0.0, 0.0]])



if __name__ == '__main__':
    x = torch.rand(100, 30)
    problem = MAF1()

    y = problem.evaluate(x)
    print()

