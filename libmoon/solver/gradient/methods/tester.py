from gradhv_solver import GradHVSolver
from epo_solver import EPOSolver
from libmoon.problem.synthetic.distribution import MOKL
import numpy as np
from torch import Tensor
import torch
import argparse
from matplotlib import pyplot as plt

from libmoon.util.constant import root_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--n-problem', type=int, default=5)
    parser.add_argument('--n-epoch', type=int, default=6000)
    parser.add_argument('--step-size', type=float, default=1e-4)
    parser.add_argument('--solver-name', type=str, default='hvmax')
    args = parser.parse_args()
    problem_name = "MOKL"
    mu_arr = [Tensor([1, 2]), Tensor([2, 3])]
    Sigma_arr = [Tensor(np.array([[1, 0.5], [0.5, 1]])),
                 Tensor(np.array([[1, 0], [0, 1]]))]
    problem = MOKL(mu_arr, Sigma_arr)
    prefs = torch.randn(args.n_problem, args.n_obj)

    if args.solver_name == 'hvmax':
        solver = GradHVSolver(prefs=prefs, step_size=args.step_size, n_epoch=args.n_epoch,
                              tol=1e-4, problem_name=problem_name, problem=problem, verbose=True)
    else:
        solver = EPOSolver(prefs=prefs, step_size=args.step_size, n_epoch=args.n_epoch,
                              tol=1e-4, problem_name=problem_name, problem=problem, verbose=True)

    x_init = torch.rand(args.n_problem, args.n_obj)
    x_init = x_init / torch.sum(x_init, axis=1, keepdim=True)
    print('Solving...')
    res = solver.solve( x_init )
    print('Solving over.')
    print('res [x]', res['x'])
    print('res [y]', res['y'])
    plt.subplot(2, 1, 1)

    if 'hv_history' in res:
        plt.plot(range(len(res['hv_history'])),
                 res['hv_history'])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    else:
        print("Warning: 'hv_history' not found in res")

    plt.subplot(2, 1, 2)
    if 'y' in res and isinstance(res['y'], (list, tuple, np.ndarray)) and len(res['y'].shape) == 2 and res['y'].shape[
        1] >= 2:
        plt.scatter(res['y'][:, 0], res['y'][:, 1])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    else:
        print("Warning: 'y' does not have the correct shape")
