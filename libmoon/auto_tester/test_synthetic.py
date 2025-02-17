# It is used to test all synthetic problems. CI.
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
libmoon_dir = os.path.dirname(os.path.dirname(current_dir))
# 将 libmoon 路径添加到 sys.path
sys.path.append(libmoon_dir)

from libmoon.solver.gradient.methods.base_solver import GradAggSolver
from libmoon.solver.gradient.methods.epo_solver import EPOSolver
from libmoon.solver.gradient.methods.mgda_solver import MGDAUBSolver
from libmoon.solver.gradient.methods.pmgda_solver import PMGDASolver
from libmoon.solver.gradient.methods.moosvgd_solver import MOOSVGDSolver
from libmoon.solver.gradient.methods.gradhv_solver import GradHVSolver
from libmoon.solver.gradient.methods.pmtl_solver import PMTLSolver

from libmoon.problem.synthetic.zdt import ZDT1
from libmoon.problem.synthetic.vlmop import VLMOP1
from libmoon.util import get_uniform_pref, get_x_init
from matplotlib import pyplot as plt
from time import time
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epoch', type=int, default=2000)
    # Tested. GradAggSolver, EPOSolver, MGDAUBSolver, PMGDASolver, MOOSVGDSolver
    parser.add_argument('--solver-name', type=str, default='PMTL')

    solver_dict = {
        'PMGDA': PMGDASolver,
        'EPO': EPOSolver,
        'MOOSVGD': MOOSVGDSolver,
        'GradHV': GradHVSolver,
        'PMTL': PMTLSolver,
        'GradAgg': GradAggSolver,
        'MGDAUB': MGDAUBSolver
    }

    solver = solver_dict[parser.parse_args().solver_name]
    args = parser.parse_args()
    problem = VLMOP1(n_var=10)
    n_probs = 10
    prefs = get_uniform_pref(n_probs, problem.n_obj, clip_eps=0.1)
    solver = solver_dict[args.solver_name](problem, prefs, n_epoch=args.n_epoch, step_size=1e-3, tol=1e-3)
    x_init = get_x_init(n_probs, problem.n_var, lbound=problem.lbound, ubound=problem.ubound)
    ts = time()
    res = solver.solve(x_init=x_init)
    ts = time() - ts
    y = res['y']
    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes[0].scatter(y[:, 0], y[:, 1])
    axes[0].set_title(solver.solver_name, fontsize=20)
    axes[0].set_xlabel('$f_1$', fontsize=20)
    axes[0].set_ylabel('$f_2$', fontsize=20)
    # Line plot in the second subplot (axes[1])
    axes[1].plot(res['hv_history'], linewidth=2)
    axes[1].set_title('HV History', fontsize=20)
    axes[1].set_xlabel('Epoch', fontsize=20)
    axes[1].set_ylabel('Hypervolume', fontsize=20)
    # Show the plot
    plt.tight_layout()
    print('elapsed :{:.2f}'.format(ts / 5000) )
    plt.show()