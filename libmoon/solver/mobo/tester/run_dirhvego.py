from libmoon.problem.synthetic.zdt import ZDT2
import matplotlib.pyplot as plt
import numpy as np
import time
from libmoon.solver.mobo.methods.dirhvego_solver import DirHVEGOSolver


if __name__ == '__main__':
    MAX_FE = 200
    BATCH_SIZE = 5  # batch size
    problem =  ZDT2(n_var=8,n_obj=2)
    n_init = 11*problem.n_var-1
    solver = DirHVEGOSolver(problem, n_init, MAX_FE, BATCH_SIZE)
    ts = time.time()
    res = solver.solve()
    elapsed = time.time() - ts
    res['elapsed'] = elapsed

    use_fig = True
    if use_fig:
        fig = plt.figure()
        plt.scatter(res['y'][res['idx_nds'][0],0], res['y'][res['idx_nds'][0],1], label='Solutions')
        plt.plot(problem.get_pf()[:,0], problem.get_pf()[:,1], label='PF')

        plt.legend(fontsize=16)
        plt.xlabel('$f_1$', fontsize=18)
        plt.ylabel('$f_2$', fontsize=18)

        plt.show()