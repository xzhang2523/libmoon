import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
libmoon_dir = os.path.dirname(os.path.dirname(current_dir))
# 将 libmoon 路径添加到 sys.path
sys.path.append(libmoon_dir)

import numpy as np
import argparse
import torch
from libmoon.util.prefs import get_uniform_pref
from libmoon.util.constant import root_name
from libmoon.util.mtl import get_dataset, model_from_dataset
from libmoon.util.network import numel
from libmoon.solver.gradient.methods.core.core_mtl import GradBaseMTLSolver, GradBasePSLMTLSolver

from libmoon.solver.gradient.methods.epo_solver import EPOCore
from libmoon.solver.gradient.methods.random_solver import RandomCore
from libmoon.solver.gradient.methods.base_solver import AggCore
from libmoon.solver.gradient.methods.core.core_mtl import GradBaseMTLSolver
from libmoon.solver.gradient.methods.mgda_solver import MGDAUBCore
from libmoon.solver.gradient.methods.pmgda_solver import PMGDACore
from libmoon.solver.gradient.methods.moosvgd_solver import MOOSVGDCore
from libmoon.solver.gradient.methods.gradhv_solver import GradHVCore
from libmoon.solver.gradient.methods.pmtl_solver import PMTLCore


# MGDAUBCore, RandomCore, AggCore, MOOSVGDCore, HVGradCore, PMTLCore)
# from libmoon.solver.gradient.methods.core.core_solver import PMGDACore
from libmoon.util.mtl import get_mtl_prefs
import os
from matplotlib import pyplot as plt
from libmoon.util.constant import plt_2d_tickle_size, plt_2d_marker_size, plt_2d_label_size
from libmoon.problem.mtl.objectives import from_name


def plot_fig_2d(folder_name, loss, prefs):
    rho = np.max([np.linalg.norm(elem) for elem in loss])
    prefs_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    plt.xlabel('$L_1$', fontsize=plt_2d_label_size)
    plt.ylabel('$L_2$', fontsize=plt_2d_label_size)
    plt.xticks(fontsize=plt_2d_tickle_size)
    plt.yticks(fontsize=plt_2d_tickle_size)
    for pref in prefs_l2:
        plt.plot([0, rho * pref[0]], [0, rho * pref[1]], color='grey', linestyle='--', linewidth=2)
    plt.scatter(loss[:, 0], loss[:, 1])
    file_name = os.path.join(folder_name, 'res.pdf')
    plt.savefig(file_name, bbox_inches='tight')
    print('Save to {}'.format(file_name))
    if args.use_plt == 'True':
        plt.show()

def save_pickle(folder_name):
    import pickle
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='M1')
    # mgdaub random epo pmgda agg_ls agg_tche agg_pbi agg_cosmos, agg_softtche pmtl hvgrad moosvgd
    parser.add_argument('--problem-name', type=str, default='adult')
    parser.add_argument('--solver-name', type=str, default='GradAgg')
    parser.add_argument('--use-plt', type=str, default='True')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--seed-idx', type=int, default=0)
    parser.add_argument('--n-prob', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--step-size', type=float, default=1e-4)
    args = parser.parse_args()
    np.random.seed(args.seed_idx)
    print('Running {} on {} with seed {}'.format(args.solver_name, args.problem_name, args.seed_idx))
    print('Using GPU') if torch.cuda.is_available() else print('Using CPU')
    model = model_from_dataset(args.problem_name)
    num_param = numel(model)
    print('Number of parameters: {}'.format(num_param))
    prefs = get_mtl_prefs(problem_name=args.problem_name, n_prob=args.n_prob)
    if args.solver_name == 'EPO':
        core_solver = EPOCore(n_var=num_param, prefs=prefs)
    elif args.solver_name == 'MGDAUB':
        core_solver = MGDAUBCore()
    elif args.solver_name == 'Random':
        core_solver = RandomCore(n_var=num_param, prefs=prefs)
    elif args.solver_name == 'PMGDA':
        core_solver = PMGDACore(n_var=num_param, prefs=prefs)
    elif args.solver_name == 'GradAgg':
        core_solver = AggCore(n_var=num_param, prefs=prefs, solver_name=args.solver_name)
    elif args.solver_name == 'MOOSVGD':
        core_solver = MOOSVGDCore(n_var=num_param, prefs=prefs)
    elif args.solver_name == 'GradHV':
        core_solver = GradHVCore(n_obj=2, n_var=num_param, problem_name=args.problem_name)
    elif args.solver_name == 'PMTL':
        core_solver = PMTLCore(n_obj=2, n_var=num_param, n_epoch=args.epoch, prefs=prefs)
    else:
        assert False, 'Unknown solver'

    solver = GradBaseMTLSolver(problem_name=args.problem_name, step_size=args.step_size, epoch=args.epoch, core_solver=core_solver,
                               batch_size=args.batch_size, prefs=prefs)

    res = solver.solve()
    res['prefs'] = prefs
    res['y'] = res['loss']
    loss = res['loss']

    folder_name = os.path.join(root_name, 'Output', 'discrete', args.problem_name, args.solver_name,
                               'seed_{}'.format(args.seed_idx))
    os.makedirs(folder_name, exist_ok=True)
    plot_fig_2d(folder_name=folder_name, loss=loss, prefs=prefs)
    save_pickle(folder_name=folder_name)