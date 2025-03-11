# It is used to test all synthetic problems. CI.
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
libmoon_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(libmoon_dir)

from libmoon.solver.gradient.methods.base_solver import GradAggSolver
from libmoon.solver.gradient.methods.epo_solver import EPOSolver
from libmoon.solver.gradient.methods.mgda_solver import MGDAUBSolver
from libmoon.solver.gradient.methods.pmgda_solver import PMGDASolver
from libmoon.solver.gradient.methods.moosvgd_solver import MOOSVGDSolver
from libmoon.solver.gradient.methods.gradhv_solver import GradHVSolver
from libmoon.solver.gradient.methods.pmtl_solver import PMTLSolver
from libmoon.solver.gradient.methods.random_solver import RandomSolver
from libmoon.solver.gradient.methods.umod_solver import UMODSolver
from libmoon.problem.synthetic.vlmop import VLMOP1
from libmoon.util import get_uniform_pref, get_x_init
from matplotlib import pyplot as plt
from time import time
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epoch', type=int, default=10000)
    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('--solver-name', type=str, default='PMGDA')
    parser.add_argument('--agg-name', type=str, default='LS')

    solver_dict = {
        'PMGDA': PMGDASolver,
        'EPO': EPOSolver,
        'MOOSVGD': MOOSVGDSolver,
        'GradHV': GradHVSolver,
        'PMTL': PMTLSolver,
        'GradAgg': GradAggSolver,
        'MGDAUB': MGDAUBSolver,
        'Random': RandomSolver,
        'UMOD': UMODSolver,
    }

    solver = solver_dict[parser.parse_args().solver_name]
    args = parser.parse_args()
    args.method_name = args.solver_name if args.solver_name != 'GradAgg' \
        else '{}_{}'.format(args.solver_name, args.agg_name)
    problem = VLMOP1(n_var=10)
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_name = os.path.join(root_path, 'Output', 'finite',
                               problem.problem_name, args.solver_name)
    os.makedirs(folder_name, exist_ok=True)

    n_probs = 10
    prefs = get_uniform_pref(n_probs, problem.n_obj, clip_eps=0.02)
    solver = solver_dict[args.solver_name](problem=problem, prefs=prefs, n_epoch=args.n_epoch,
                                           step_size=args.step_size, tol=1e-3, folder_name=folder_name)

    x_init = get_x_init(n_probs, problem.n_var,
                        lbound=problem.lbound, ubound=problem.ubound)

    ts = time()
    res = solver.solve(x_init=x_init)
    ts = time() - ts
    y = res['y']
    fig = plt.figure()
    plt.scatter(y[:, 0], y[:, 1], s=100)
    rho_arr = np.linalg.norm(y, axis=1)
    plt.xlabel('$f_1$', fontsize=20)
    plt.ylabel('$f_2$', fontsize=20)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.gca().set_aspect('equal', adjustable="box")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    os.makedirs(folder_name, exist_ok=True)
    prefs = prefs.numpy()
    prefs_norm = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    rho_max = 1.2
    for pref in prefs_norm:
        plt.plot([0, pref[0] * rho_max ], [0, pref[1] * rho_max ],
                 color='k', linewidth=2, linestyle='--')
    plt.plot(problem.get_pf()[:,0], problem.get_pf()[:,1], color='r', linewidth=1, linestyle='--')
    fig_name = os.path.join(folder_name, '{}.pdf'.format(args.method_name) )
    plt.savefig(fig_name, bbox_inches='tight')
    print('Saved in {}'.format(fig_name))

    fig = plt.figure()
    plt.plot(res['hv_history'], linewidth=2)
    plt.title('HV History', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Hypervolume', fontsize=20)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    print('elapsed :{:.2f}'.format(ts / 5000) )
    fig_name = os.path.join(folder_name, 'HV_{}.pdf'.format(args.method_name) )
    plt.savefig(fig_name, bbox_inches='tight')
    print('HV file Saved in {}'.format(fig_name))

