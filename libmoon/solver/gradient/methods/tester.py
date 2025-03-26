from stack_data import Variable

from gradhv_solver import GradHVSolver
from libmoon.problem.synthetic.distribution import MOKL
import numpy as np
from torch import Tensor
import torch
from torch.autograd import Variable


if __name__ == '__main__':

    problem_name = "MOKL"
    mu_arr = [Tensor([1, 2]), Tensor([2, 3])]
    Sigma_arr = [Tensor(np.array([[1, 0.5], [0.5, 1]])),
                 Tensor(np.array([[1, 0], [0, 1]]))]

    problem = MOKL(mu_arr, Sigma_arr)
    n_obj = 2
    n_problems = 10
    prefs = torch.randn(n_problems, n_obj)
    solver = GradHVSolver(prefs=prefs, step_size=1e-3, n_epoch=100,
                          tol=1e-4, problem_name=problem_name, problem=problem)

    x_init = Variable(torch.randn(2))
    x_init = x_init / torch.sum(x_init)
    res = solver.solve( x_init )
    print(res)

