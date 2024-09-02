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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--FE', type=int, default=200)
    args = parser.parse_args()
    # minimization
    problem = ZDT1(n_var=8,n_obj=2)
    n_init = 11*problem.n_var-1
    ts = time.time()
    solver_lcb = PSLMOBOSolver(problem, n_init, args.FE, args.batch_size)
    res = solver_lcb.solve()
    elapsed = time.time() - ts
    res['elapsed'] = elapsed

    use_fig = True
    if use_fig:
        fig = plt.figure()
        plt.scatter(res['y'][res['idx_nds'][0],0], res['y'][res['idx_nds'][0],1], label='Solutions')
        plt.plot(problem._get_pf()[:,0], problem._get_pf()[:,1], label='PF')

        plt.legend(fontsize=16)
        plt.xlabel('$f_1$', fontsize=18)
        plt.ylabel('$f_2$', fontsize=18)

        folder_name = 'D:\\pycharm_project\\libmoon\\Output\\mobo\\pslmobo'
        os.makedirs(folder_name, exist_ok=True)
        fig_name = os.path.join(folder_name, 'ZDT1.pdf')
        print('Saved in {}'.format(fig_name) )
        plt.savefig(fig_name)
        plt.show()











