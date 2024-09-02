"""
    Runing the psl+DirHV-EI method on ZDT1
"""
import sys
sys.path.append('D:\\pycharm_project\\libmoon')

import torch
import time
from libmoon.solver.mobo.methods.psldirhvei_solver import (PSLDirHVEISolver)
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
from libmoon.problem.synthetic.zdt import ZDT1
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--FE', type=int, default=20)
    args = parser.parse_args()
    # MAX_FE = 200
    # BATCH_SIZE = 5  # batch size
    # minimization
    problem =  ZDT1(n_var=8,n_obj=2)
    n_init = 11*problem.n_var-1
    ts = time.time()
    solver_ei = PSLDirHVEISolver(problem, n_init, args.FE, args.batch_size)
    res = solver_ei.solve()
    elapsed = time.time() - ts
    res['elapsed'] = elapsed

    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\Output', 'mobo', solver_ei.solver_name)
    os.makedirs(folder_name, exist_ok=True)

    use_fig = True
    if use_fig:
        fig = plt.figure()
        plt.scatter(res['y'][res['idx_nds'][0],0], res['y'][res['idx_nds'][0],1], label='Solutions')
        plt.plot(problem._get_pf()[:,0], problem._get_pf()[:,1], label='PF')
        plt.legend(fontsize=16)
        plt.xlabel('$f_1$', fontsize=18)
        plt.ylabel('$f_2$', fontsize=18)
        fig_name = os.path.join(folder_name, 'ZDT1.pdf')
        plt.savefig(fig_name)
        print('Saved in {}'.format(fig_name) )
        plt.show()

   
    
 
 
 
       
    
 
 

