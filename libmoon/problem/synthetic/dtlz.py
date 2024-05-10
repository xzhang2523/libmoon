
import numpy as np
import torch
from libmoon.problem.synthetic.mop import mop




class DTLZ1(mop):

    def __init__(self, n_var=30, n_obj=3, lbound=np.zeros(30),
                 ubound=np.ones(30)):

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound)
        self.problem_name= 'DTLZ1'

    def _evaluate_torch(self, x: torch.Tensor):
        x1 = x[:,0]
        x2 = x[:,1]
        xm = x[:, 2:]

        g = 100 * (self.n_var - 2 + torch.sum(torch.pow(xm - 0.5, 2) -
                                              torch.cos(20 * np.pi * (xm - 0.5)), dim=1))

        f1 = 0.5 * x1 * x2 * (1+g)
        f2 = 0.5 * x1 * (1 - x2) * (1+g)
        f3 = 0.5 * (1 - x1) * (1+g)
        return torch.stack((f1, f2, f3), dim=1)

    def _evaluate_numpy(self, x: np.ndarray):
        x1 = x[:, 0]
        x2 = x[:, 1]
        xm = x[:, 2:]

        g = 100 * (self.n_var - 2 + np.sum(np.power(xm - 0.5, 2) -
                                              np.cos(20 * np.pi * (xm - 0.5)), axis=1))

        f1 = 0.5 * x1 * x2 * (1+g)
        f2 = 0.5 * x1 * (1 - x2) * (1+g)
        f3 = 0.5 * (1 - x1) * (1+g)
        return np.stack((f1, f2, f3), axis=1)



class DTLZ2(mop):
    def __init__(self, n_var=30, n_obj=3, lbound=np.zeros(30),
                 ubound=np.ones(30)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'DTLZ2'

    def _evaluate_torch(self, x):
        xm = x[:, 2:]
        g = torch.sum(torch.pow(xm - 0.5, 2), dim=1)
        f1 = torch.cos(x[:, 0] * np.pi / 2) * torch.cos(x[:, 1] * np.pi / 2) * (1 + g)
        f2 = torch.cos(x[:, 0] * np.pi / 2) * torch.sin(x[:, 1] * np.pi / 2) * (1 + g)
        f3 = torch.sin(x[:, 0] * np.pi / 2) * (1 + g)
        return torch.stack((f1, f2, f3), dim=1)

    def _evaluate_numpy(self, x):
        xm = x[:, 2:]
        g = np.sum(np.power(xm - 0.5, 2), axis=1)
        f1 = np.cos(x[:, 0] * np.pi / 2) * np.cos(x[:, 1] * np.pi / 2) * (1 + g)
        f2 = np.cos(x[:, 0] * np.pi / 2) * np.sin(x[:, 1] * np.pi / 2) * (1 + g)
        f3 = np.sin(x[:, 0] * np.pi / 2) * (1 + g)
        return np.stack((f1, f2, f3), axis=1)


class DTLZ3(mop):
    def __init__(self, n_var=30, n_obj=3, lbound=np.zeros(30),
                 ubound=np.ones(30)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'DTLZ3'


    def _evaluate_torch(self, x):
        xm = x[:, 2:]
        g = 100 * (self.n_var - 2 + torch.sum(torch.pow(xm - 0.5, 2) -
                                              torch.cos(20 * np.pi * (xm - 0.5)), dim=1))
        f1 = torch.cos(x[:, 0] * np.pi / 2) * torch.cos(x[:, 1] * np.pi / 2) * (1 + g)
        f2 = torch.cos(x[:, 0] * np.pi / 2) * torch.sin(x[:, 1] * np.pi / 2) * (1 + g)
        f3 = torch.sin(x[:, 0] * np.pi / 2) * (1 + g)
        return torch.stack((f1, f2, f3), dim=1)

    def _evaluate_numpy(self, x):
        xm = x[:, 2:]

        g = 100 * (self.n_var - 2 + np.sum(np.power(xm - 0.5, 2) -
                                              np.cos(20 * np.pi * (xm - 0.5)), axis=1))

        f1 = np.cos(x[:, 0] * np.pi / 2) * np.cos(x[:, 1] * np.pi / 2) * (1 + g)
        f2 = np.cos(x[:, 0] * np.pi / 2) * np.sin(x[:, 1] * np.pi / 2) * (1 + g)
        f3 = np.sin(x[:, 0] * np.pi / 2) * (1 + g)
        return np.stack((f1, f2, f3), axis=1)


class DTLZ4(mop):
    def __init__(self, n_var=30, n_obj=3, lbound=np.zeros(30),
                 ubound=np.ones(30)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'DTLZ4'
        self.alpha = 20

    def _evaluate_torch(self, x):
        xm = x[:, 2:]
        g = torch.sum(torch.pow(xm - 0.5, 2), dim=1)
        # alpha = 1

        f1 = torch.cos(x[:, 0] ** self.alpha * np.pi / 2) * torch.cos(x[:, 1] ** self.alpha * np.pi / 2) * (1 + g)
        f2 = torch.cos(x[:, 0] ** self.alpha * np.pi / 2) * torch.sin(x[:, 1] ** self.alpha * np.pi / 2) * (1 + g)
        f3 = torch.sin(x[:, 0] ** self.alpha * np.pi / 2) * (1 + g)
        return torch.stack((f1, f2, f3), dim=1)

    def _evaluate_numpy(self, x):
        xm = x[:, 2:]
        g = np.sum(np.power(xm - 0.5, 2), axis=1)

        f1 = np.cos(x[:, 0] ** self.alpha * np.pi / 2) * np.cos(x[:, 1] ** self.alpha * np.pi / 2) * (1 + g)
        f2 = np.cos(x[:, 0] ** self.alpha * np.pi / 2) * np.sin(x[:, 1] ** self.alpha * np.pi / 2) * (1 + g)
        f3 = np.sin(x[:, 0] ** self.alpha * np.pi / 2) * (1 + g)
        return np.stack((f1, f2, f3), axis=1 )

# DTLZ5, DTLZ6.
# degenerated.


# DTLZ7 has disjoint Pareto front.



if __name__ == '__main__':
    x = torch.rand(100, 30)
    problem = DTLZ4()

    y = problem.evaluate(x)
    print( y )



