"""
    Runing the psl+lcb method on ZDT1
"""
import os.path
import sys
sys.path.append('D:\\pycharm_project\\libmoon')

import torch
import time
from libmoon.solver.mobo.methods.pslmobo_solver import PSLMOBOSolver
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
from libmoon.problem.synthetic.zdt import ZDT1
import argparse
from libmoon.util import random_everything, save_pickle
from libmoon.util.problems import get_problem
from libmoon.solver.mobo.utils.lhs import lhs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--FE', type=int, default=200)
    parser.add_argument('--n-var', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--problem-name', type=str, default='ZDT1')
    parser.add_argument('--use-fig', type=str, default='True')

    args = parser.parse_args()
    random_everything(args.seed)
    print('seed: {} on problem: {}'.format(args.seed, args.problem_name) )

    # minimization
    problem = get_problem(args.problem_name, n_var=args.n_var)
    n_init = 11*problem.n_var-1
    ts = time.time()

    x_init = torch.from_numpy(lhs(problem.n_var, samples=n_init))
    solver = PSLMOBOSolver(problem, x_init, args.FE, args.batch_size)
    res = solver.solve()
    elapsed = time.time() - ts
    res['elapsed'] = elapsed

    fig = plt.figure()
    plt.scatter(res['y'][res['idx_nds'][0],0], res['y'][res['idx_nds'][0],1], label='Solutions')
    if hasattr(problem, '_get_pf'):
        plt.plot(problem._get_pf()[:,0], problem._get_pf()[:,1], label='PF')

    plt.legend(fontsize=16)
    plt.xlabel('$f_1$', fontsize=18)
    plt.ylabel('$f_2$', fontsize=18)

    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\Output\\mobo', args.problem_name,solver.solver_name, 'seed_{}'.format(args.seed))


    os.makedirs(folder_name, exist_ok=True)
    fig_name = os.path.join(folder_name, 'res.pdf')
    print('Saved in {}'.format(fig_name) )
    plt.savefig(fig_name)
    if args.use_fig == 'True':
        plt.show()

    pickle_name = os.path.join(folder_name, '{}.pickle'.format('res'))
    save_pickle(res, pickle_name)
    print('Saved in {}'.format(pickle_name) )












