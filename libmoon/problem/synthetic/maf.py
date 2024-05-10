
from numpy import array
import torch



class MAF1:
    def __init__(self, n_var):
        '''
            n_obj can be set as any number. For simlicity, we set it as 3.
        '''
        self.n_obj = 3
        self.n_var = n_var
        self.lb = 0
        self.ub = 1

    def evaluate(self, x):
        if type(x) == torch.Tensor:

            g = torch.sum( torch.pow(x[:, 2:] - 0.5, 2), dim=1 )

            f1 = (1 - x[:,0] * x[:,1]) * (1 + g)
            f2 = (1 - x[:,0] * (1 - x[:,1]) ) * (1 + g)
            f3 = x[:,0] * (1 + g)

            return torch.stack((f1, f2, f3), dim=1)

        else:
            assert False



    def get_pf(self):
        return array([[0.0, 0.0, 0.0]])



if __name__ == '__main__':
    x = torch.rand(100, 30)
    problem = MAF1()

    y = problem.evaluate(x)
    print()

