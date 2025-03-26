import torch
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from torch import Tensor
import numpy as np
from libmoon.problem.synthetic.zdt import ZDT1
from libmoon.solver.gradient.methods.core.core_solver import RandomCore

class CoreRandom:
    def __init__(self, args):
        self.args = args

    def get_weight(self):
        return Tensor(np.random.rand(10, 2))

class RandomSolver(GradBaseSolver):
    def __init__(self, step_size, n_iter, tol, problem, prefs):
        self.step_size = step_size
        self.n_iter = n_iter
        self.tol = tol
        self.problem = problem
        self.prefs = prefs
        self.solver_cls = RandomCore()

    def solve(self, x):
        return super().solve(self.problem, x, self.prefs, self.solver_cls)

if __name__ == '__main__':
    problem = ZDT1(n_var=10)
    solver = RandomSolver(0.1, 100, 1e-6)
    x = torch.rand((10,10))
    pref_1d = torch.linspace(0, 1, 10)
    prefs = torch.stack((pref_1d, 1 - pref_1d), dim=1)
    res = solver.solve(problem, x, prefs)
