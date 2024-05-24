
import argparse

import numpy as np

from libmoon.solver.gradient.methods import EPOSolver
from libmoon.util_global.initialization import synthetic_init
from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global.problems import get_problem
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt




if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script' )
    parser.add_argument( '--n-partition', type=int, default=10 )

    # Inv agg not very work
    parser.add_argument( '--agg', type=str, default='tche')  # If solve is agg, then choose a specific agg method.
    parser.add_argument('--solver', type=str, default='epo')
    parser.add_argument( '--problem-name', type=str, default='VLMOP2')

    parser.add_argument('--n-iter', type=int, default=1000 )
    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--plt-pref-flag', type=str, default='N')
    parser.add_argument('--use-plt', type=str, default='Y')
    # For Pmgda
    parser.add_argument('--h-eps', type=float, default=1e-2)
    parser.add_argument('--sigma', type=float, default=0.8)
    args = parser.parse_args()

    # problem_name = 'ZDT1'

    color_arr = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']*10
    hv_seed = []
    for seed_idx in range(1):

        # The four key codes.
        problem = get_problem(problem_name='DTLZ2')
        prefs = uniform_pref(n_prob=20, n_obj = problem.n_obj, clip_eps=1e-2)
        solver = EPOSolver(problem, step_size=1e-2, n_iter=args.n_iter, tol=1e-2)
        res = solver.solve( x=synthetic_init(problem, prefs), prefs=prefs )

        ax = (plt.figure()).add_subplot(projection='3d')
        for idx in range(5) :
            ax.plot(res['y_history'][:, idx, 0], res['y_history'][:, idx, 1], res['y_history'][:, idx, 2], color=color_arr[idx])

        th1 = np.linspace(0, np.pi/2, 100)
        th2 = np.linspace(0, np.pi/2, 100)
        theta, phi = np.meshgrid(th1, th2)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        ax.plot_surface(x, y, z, alpha=0.3)

        ax.view_init(30, 45)


        hv_seed.append( res['hv_history'] )



    # hv_mean = np.mean( np.array(hv_seed), axis=0 )
    # hv_std = np.std( np.array(hv_seed), axis=0 )
    # plt.plot( hv_mean )
    # plt.fill_between( range(args.n_iter), hv_mean-hv_std, hv_mean+hv_std, alpha=0.3 )
    plt.show()