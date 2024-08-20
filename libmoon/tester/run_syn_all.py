import sys
sys.path.append('D:\\pycharm_project\\libmoon')


from libmoon.solver.gradient.methods import MGDAUBSolver, EPOSolver, RandomSolver
from libmoon.problem.synthetic.zdt import ZDT1
from libmoon.util_global.constant import root_name

import argparse
import torch
import pickle
import os



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default='random')
    parser.add_argument('--problem', type=str, default='ZDT1')
    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--n-iter', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tol', type=float, default=1e-2)
    args = parser.parse_args()


    n_prob = 10
    n_var = 10
    pref1d = torch.linspace(0.1, 0.9, n_prob)
    prefs = torch.stack([pref1d, 1 - pref1d], dim=1)

    problem = ZDT1(n_var=n_var)
    if args.solver == 'epo':
        solver = EPOSolver(step_size=1e-2, n_iter=args.n_iter, tol=1e-2, problem=problem, prefs=prefs)
    elif args.solver == 'mgdaub':
        solver = MGDAUBSolver(step_size=1e-2, n_iter=args.n_iter, tol=1e-2, problem=problem, prefs=prefs)
    elif args.solver == 'random':
        solver = RandomSolver(step_size=1e-2, n_iter=args.n_iter, tol=1e-2, problem=problem, prefs=prefs)
    else:
        assert False, 'Unknown solver {}'.format(args.solver)
    res = solver.solve(x=torch.rand(n_prob, n_var))

    folder_name = os.path.join(root_name, 'Output', args.solver, args.problem)
    os.makedirs(folder_name, exist_ok=True)
    pickle_name = os.path.join(folder_name, 'res.pkl')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Result saved to {}'.format(pickle_name))


