import numpy as np
import argparse
import torch

from libmoon.util.prefs import uniform_pref
from libmoon.util.constant import root_name
from libmoon.util.mtl import get_dataset, model_from_dataset
from libmoon.util.network import numel

from libmoon.solver.gradient.methods.core.core_pref_mtl import GradBaseMTLSolver
from libmoon.solver.gradient.methods.core.core_solver import EPOCore
import os





def plot_fig_2d(folder_name):
    # fig = plt.figure()
    # for idx, res_elem in enumerate(res):
    #     plt.scatter(res_elem[0], res_elem[1], color=color_arr[idx], label='Pref. {}'.format(idx + 1))
    # rho = np.max(np.linalg.norm(res, axis=1))
    # for idx, pref in enumerate(pref_mat):
    #     plt.plot([0, pref[0] * rho], [0, pref[1] * rho], color=color_arr[idx])
    # plt.xlabel('$L_1$', fontsize=18)
    # plt.ylabel('$L_2$', fontsize=18)
    # plt.legend(fontsize=12)
    # fig_name = os.path.join(folder_name, '{}.pdf'.format('fig'))
    # plt.savefig(fig_name)
    # print('Save fig to {}'.format(fig_name))
    pass


def save_pickle(folder_name):
    # pickle_file_name = os.path.join(folder_name, 'res.pickle')
    #
    # with open(pickle_file_name, 'wb') as f:
    #     pickle.dump({
    #         'res': res,
    #     }, f)
    # print('Save pickle to {}'.format(pickle_file_name))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='M1')
    # mgdaub random epo pmgda agg_ls agg_tche agg_pbi agg_cosmos, agg_softtche pmtl hvgrad moosvgd
    parser.add_argument('--problem-name', type=str, default='adult')
    parser.add_argument('--solver-name', type=str, default='epo')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--seed-idx', type=int, default=0)
    parser.add_argument('--n-prob', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--step-size', type=float, default=1e-3)
    args = parser.parse_args()

    np.random.seed(args.seed_idx)
    print('Running {} on {} with seed {}'.format(args.solver_name, args.problem_name, args.seed_idx))
    print('Using GPU') if torch.cuda.is_available() else print('Using CPU')

    model = model_from_dataset(args.problem_name)
    num_param = numel(model)
    print('Number of parameters: {}'.format(num_param))

    prefs = uniform_pref(n_prob=args.n_prob, n_obj = 2, clip_eps=1e-2)
    if args.solver_name == 'epo':
        core_solver = EPOCore(n_var=num_param, prefs=prefs)

    solver = GradBaseMTLSolver(n_prob=args.n_prob, problem_name=args.problem_name, step_size=args.step_size, epoch=args.epoch, core_solver=core_solver,
                               batch_size=args.batch_size)

    # pref_mat = get_mtl_prefs(args.dataset_name, kwargs['n_prob'], obj_normalization=kwargs['obj_normalization'])
    # res = solver.solve(pref_mat)

    folder_name = os.path.join(root_name, 'Output', 'discrete', args.problem_name, args.solver_name,
                               'seed_{}'.format(args.seed_idx))
    os.makedirs(folder_name, exist_ok=True)
    plot_fig_2d()
    save_pickle()




