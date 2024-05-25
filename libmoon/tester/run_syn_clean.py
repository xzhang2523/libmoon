
import argparse
import numpy as np
from libmoon.solver.gradient.methods import EPOSolver
from libmoon.util_global.weight_factor import uniform_pref

# from libmoon.util_global.initialization import synthetic_init
# from libmoon.util_global.problems import get_problem
from libmoon.util_global import synthetic_init, get_problem, uniform_pref

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt





if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script' )
    parser.add_argument( '--n-partition', type=int, default=10 )

    # Inv agg not very work
    parser.add_argument( '--agg', type=str, default='tche')  # If solve is agg, then choose a specific agg method.
    parser.add_argument('--solver', type=str, default='epo')
    parser.add_argument( '--problem-name', type=str, default='DTLZ2')

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
    color_arr = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'] * 10
    hv_seed = []

    seed_num = 3
    for seed_idx in range(seed_num):

        problem = get_problem(problem_name='DTLZ2', n_var=30)

        prefs = uniform_pref(n_prob=15, n_obj = problem.n_obj, clip_eps=1e-2)
        solver = EPOSolver(problem, step_size=1e-2, n_iter=args.n_iter, tol=1e-2)
        res = solver.solve( x=synthetic_init(problem, prefs), prefs=prefs )
        hv_seed.append( res['hv_history'] )

        sub_sample=2
        ax = (plt.figure()).add_subplot(projection='3d')
        for idx in range( len(prefs) ) :
            ax.plot(res['y_history'][::sub_sample, idx, 0], res['y_history'][::sub_sample, idx, 1], res['y_history'][::sub_sample, idx, 2],
                    color=color_arr[idx])
        prefs_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
        for idx, pref in enumerate(prefs_l2):
            ax.scatter(pref[0], pref[1], pref[2], color=color_arr[idx], s=40)

        th1 = np.linspace(0, np.pi/2, 100)
        th2 = np.linspace(0, np.pi/2, 100)

        theta, phi = np.meshgrid(th1, th2)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)

        ax.plot_surface(x, y, z, alpha=0.3)
        ax.axis('equal')

        ax.view_init(30, 45)
        ax.set_xlim([0, 1.4])
        ax.set_ylim([0, 1.4])
        ax.set_zlim([0, 1.4])