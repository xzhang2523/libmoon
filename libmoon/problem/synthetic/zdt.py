"""
ZDT test suite for multi-objective problem

Reference
----------
    Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective
    evolutionary algorithms: Empirical results. Evolutionary computation,
    8(2), 173-195. DOI: 10.1162/106365600568202
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from libmoon.problem.synthetic.mop import mop

class ZDT1(mop):

    def __init__(self, n_var=30, n_obj=2):
        lbound = np.zeros(n_var)
        ubound = np.ones(n_var)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'ZDT1'


    def _evaluate_torch(self, x: torch.Tensor):
        f1 = x[:, 0]
        n = x.shape[-1]
        g = 1 + 9/(n-1) * torch.sum(x[:, 1:], dim=1)
        h = 1 - torch.sqrt(f1 / g)
        f2 = g * h
        return torch.stack((f1, f2), dim=1)

    def _evaluate_numpy(self, x: np.ndarray):
        assert len(x.shape)==2
        n = x.shape[-1]

        f1 = x[:, 0]
        g = 1 + 9 / (n-1) * np.sum(x[:, 1:], axis=1)
        f2 = 1 - np.sqrt(f1 / g)
        return np.stack((f1, f2), axis=1)


    def _get_pf(self, n_points: int = 100):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)


class ZDT2(mop):

    def __init__(self, n_var=30, n_obj=2):
        lbound = np.zeros(n_var)
        ubound = np.ones(n_var)

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound)
        self.problem_name = 'ZDT2'

    def _evaluate_torch(self, x: torch.Tensor):
        f1 = x[:, 0]
        g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
        f2 = g * (1 - (f1 / g) ** 2)
        return torch.stack((f1, f2), dim=1)

    def _evaluate_numpy(self, x: np.ndarray):
        f1 = x[:, 0]
        g = 1 + 9 * np.mean(x[:, 1:], axis=1)
        f2 = g * (1 - (f1 / g) ** 2)
        return np.stack((f1, f2), axis=1)

    def _get_pf(self, n_points: int = 100):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
        return np.stack((f1, f2), axis=1)


class ZDT3(mop):

    def __init__(self, n_var=30, n_obj=2, lbound=np.zeros(30), ubound=np.ones(30)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'ZDT3'


    def _evaluate_torch(self, x: torch.Tensor):
        f1 = x[:, 0]
        g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
        f2 = g * (1 - torch.sqrt(f1 / g) - f1 / g * torch.sin(10 * np.pi * f1))
        return torch.stack((f1, f2), dim=1)

    def _evaluate_numpy(self, x: np.ndarray):
        f1 = x[:, 0]
        g = 1 + 9 * np.mean(x[:, 1:], axis=1)
        f2 = g * (1 - np.sqrt(f1 / g) - f1 / g * np.sin(10 * np.pi * f1))
        return np.stack((f1, f2), axis=1)

    def _get_pf(self, n_points: int = 100):
        f1 = np.hstack([np.linspace(0, 0.0830, int(n_points / 5)),
                        np.linspace(0.1822, 0.2578, int(n_points / 5)),
                        np.linspace(0.4093, 0.4539, int(n_points / 5)),
                        np.linspace(0.6183, 0.6525, int(n_points / 5)),
                        np.linspace(0.8233, 0.8518, n_points - 4 * int(n_points / 5))])
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        return np.stack((f1, f2), axis=1)


class ZDT4(mop):

    def __init__(self, n_var=10, n_obj=2, lbound=-5*np.ones(10), ubound=5*np.ones(10)):
        lbound[0] = 0
        ubound[0] = 1

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'ZDT4'


    def _evaluate_torch(self, x: torch.Tensor):
        f1 = x[:, 0]
        g = 1 + 10 * (self.n_var - 1) + torch.sum(x[:, 1:] ** 2 - 10 * torch.cos(4 * np.pi * x[:, 1:]), dim=1)
        f2 = g * (1 - torch.sqrt(f1 / g))
        return torch.stack((f1, f2), dim=1)

    def _evaluate_numpy(self, x: np.ndarray):
        f1 = x[:, 0]
        g = 1 + 10 * (self.n_var - 1) + np.sum(x[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * x[:, 1:]), axis=1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.stack((f1, f2), axis=1)

    def _get_pf(self, n_points: int = 100):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
        return np.stack((f1, f2), axis=1)


class ZDT6(mop):

    def __init__(self, n_var=30, n_obj=2, lbound=np.zeros(30), ubound=np.ones(30) ) -> None:
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'ZDT6'


    def _evaluate_torch(self, x: torch.Tensor):
        f1 = 1 - torch.exp(-4 * x[:, 0]) * (torch.sin(6 * np.pi * x[:, 0])) ** 6
        g = 1 + 9 * (torch.sum(x[:, 1:], dim=1) / (self.n_var - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return torch.stack((f1, f2), dim=1)

    def _evaluate_numpy(self, x: np.ndarray):
        f1 = 1 - np.exp(-4 * x[:, 0]) * (np.sin(6 * np.pi * x[:, 0])) ** 6
        g = 1 + 9 * (np.sum(x[:, 1:], axis=1) / (self.n_var - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return np.stack((f1, f2), axis=1)

    def _get_pf(self, n_points: int = 100):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
        return np.stack((f1, f2), axis=1)



if __name__ == '__main__':
    problem = ZDT3()

    res = problem.evaluate(torch.rand(10, problem.get_number_variable))
    pf = problem.get_pf()
    x = np.random.rand(10, problem.get_number_variable)
    y = problem.evaluate(x)

    plt.scatter(pf[:, 0], pf[:, 1], c='none', edgecolors='r')
    plt.show()
