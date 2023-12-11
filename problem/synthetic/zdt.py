import numpy as np
import torch
from torch import Tensor
import argparse
from matplotlib import pyplot as plt

class ZDT1:
    def __init__(self):
        self.n_var = 30
        self.n_obj = 2
        self.lb=0
        self.ub=1

    def evaluate(self, x):
        if type(x) == torch.Tensor:
            f1 = x[:, 0]
            g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
            f2 = g * (1 - torch.sqrt(f1 / g))
            return torch.stack((f1, f2), dim=1)
        else:
            f1 = x[:, 0]
            g = 1 + 9 * np.mean(x[:, 1:], axis=1)
            f2 = g * (1 - np.sqrt(f1 / g))
            return np.stack((f1, f2), axis=1)

    def get_pf(self):
        x = np.linspace(0, 1, 100)
        y = 1 - np.sqrt(x)
        return np.stack((x, y), axis=1)


class ZDT2:
    def __init__(self):
        self.n_var = 30
        self.n_obj = 2
        self.lb=0
        self.ub=1

    def evaluate(self, x):
        f1 = x[:, 0]
        g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
        f2 = g * (1 - (f1 / g)**2)
        return torch.stack((f1, f2), dim=1)

    def get_pf(self):
        x = np.linspace(0, 1, 100)
        y = 1 - x**2
        return np.stack((x, y), axis=1)


class ZDT3:
    def __init__(self):
        self.n_var = 30
        self.n_obj = 2
        self.lb=0
        self.ub=1

    def evaluate(self, x):
        f1 = x[:, 0]
        g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
        f2 = g * (1 - torch.sqrt(f1 / g) - f1 / g * torch.sin(10 * np.pi * f1))
        return torch.stack((f1, f2), dim=1)

    def get_pf(self):
        # print()
        x1 = np.linspace(0, 0.0830, 10)
        x2 = np.linspace(0.1822, 0.2578, 10)
        x3 = np.linspace(0.4093, 0.4539, 10)
        x4 = np.linspace(0.6183, 0.6525, 10)
        x5 = np.linspace(0.8233, 0.8518, 10)
        f1 = np.concatenate((x1, x2, x3, x4, x5))
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        return np.stack((f1, f2), axis=1)





class ZDT4:
    def __init__(self):
        self.n_var = 10
        self.n_obj = 2
        self.lb=-5
        self.ub=5

    def evaluate(self, x):
        f1 = x[:, 0]
        g = 1 + 10 * (self.n_var - 1) + torch.sum(x[:, 1:]**2 - 10 * torch.cos(4 * np.pi * x[:, 1:]), dim=1)
        f2 = g * (1 - torch.sqrt(f1 / g))
        return torch.stack((f1, f2), dim=1)

    def get_pf(self):
        x = np.linspace(0, 1, 100)
        y = 1 - np.sqrt(x)
        return np.stack((x, y), axis=1)


class ZDT6:
    def __init__(self):
        self.n_var = 10
        self.n_obj = 2
        self.lb=0
        self.ub=1

    def evaluate(self, x):
        f1 = 1 - torch.exp(-4 * x[:, 0]) * (torch.sin(6 * np.pi * x[:, 0]))**6
        g = 1 + 9 * (torch.sum(x[:, 1:], dim=1) / (self.n_var - 1))**0.25
        f2 = g * (1 - (f1 / g)**2)
        return torch.stack((f1, f2), dim=1)

    def get_pf(self):
        x = np.linspace(0, 1, 100)
        y = 1 - x**2
        return np.stack((x, y), axis=1)








if __name__ == '__main__':

    parser = argparse.ArgumentParser( description= 'example' )
    parser.add_argument( '--problem', type=int, default=100 )
    args = parser.parse_args()

    problem = ZDT1()

    # res = problem.evaluate(torch.rand(100, 30))
    # print()
    # pf = problem.get_pf()
    # plt.scatter(pf[:, 0], pf[:, 1], c='none', edgecolors='r')
    # plt.show()
#

    x = np.random.rand(100, 30)
    y = problem.evaluate(x)
    # print()