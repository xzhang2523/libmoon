import argparse
import numpy as np
import sys
sys.path.append('D:\\pycharm_project\\libmoon\\')
from libmoon.util_global import synthetic_init, get_problem, uniform_pref
from libmoon.solver.gradient.methods import EPOSolver
from libmoon.solver.gradient.methods.pmgda_solver import PMGDASolver
from libmoon.solver.gradient.methods.base_solver import GradAggSolver

from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.solver.gradient.methods.core.core_solver import EPOCore, MGDAUBCore, RandomCore, AggCore, MOOSVGDCore



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt
from libmoon.util_global.constant import FONT_SIZE_2D, FONT_SIZE_3D, color_arr, beautiful_dict, root_name
from libmoon.util_global.constant import plt_2d_pickle_size, plt_2d_marker_size, plt_2d_label_size

def draw_2d_prefs(prefs):
    prefs_norm2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    for idx, pref in enumerate(prefs_norm2):
        plt.plot([0, pref[0]], [0, pref[1]], color='grey', linewidth=2,
                 linestyle='--')


def plot_figure_2d(problem):
    y_arr = res['y']
    plt.scatter(y_arr[:, 0], y_arr[:, 1], color='black')
    plt.xticks(fontsize=plt_2d_pickle_size)
    plt.yticks(fontsize=plt_2d_pickle_size)
    plt.xlabel('$L_1$', fontsize=plt_2d_label_size)
    plt.ylabel('$L_2$', fontsize=plt_2d_label_size)
    plt.axis('equal')
    # plt.grid()
    pf = problem.get_pf(n_pareto_points=1000)
    plt.plot(pf[:, 0], pf[:, 1], color='red', linewidth=2, label='True PF')
    plt.legend(fontsize=15)
    draw_2d_prefs(prefs)

def plot_figure_3d():
    sub_sample = 1
    ax = (plt.figure()).add_subplot(projection='3d')
    for idx in range(len(prefs)):
        ax.plot(res['y_history'][::sub_sample, idx, 0], res['y_history'][::sub_sample, idx, 1],
                res['y_history'][::sub_sample, idx, 2],
                color=color_arr[idx])
    prefs_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    for idx, pref in enumerate(prefs_l2):
        ax.scatter(pref[0], pref[1], pref[2], color=color_arr[idx], s=40)
    th1 = np.linspace(0, np.pi / 2, 100)
    th2 = np.linspace(0, np.pi / 2, 100)

    theta, phi = np.meshgrid(th1, th2)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    ax.plot_surface(x, y, z, alpha=0.3)
    ax.axis('equal')
    ax.view_init(30, 45)
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 1.2])
    ax.set_zlim([0, 1.2])
    ax.set_xlabel('$L_1$', fontsize=FONT_SIZE_3D)
    ax.set_ylabel('$L_2$', fontsize=FONT_SIZE_3D)
    ax.set_zlabel('$L_3$', fontsize=FONT_SIZE_3D)

def save_figures(folder_name):
    # os.makedirs(folder_name, exist_ok=True)
    fig_name = os.path.join(folder_name, 'res.pdf')
    plt.savefig(fig_name, bbox_inches='tight')
    fig_name_svg = os.path.join(folder_name, 'res.svg')
    plt.savefig(fig_name_svg, bbox_inches='tight')
    print('Save fig to {}'.format(fig_name))
    plt.title(beautiful_dict[args.solver_name])


def save_pickles(folder_name):
    import pickle
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script' )
    # solver_array: []
    # solver_name: ['hv_grad', 'pmtl', '']
    parser.add_argument('--solver-name', type=str, default='moosvgd')
    parser.add_argument( '--problem-name', type=str, default='VLMOP1')
    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--draw-fig', type=str, default='True')
    parser.add_argument('--use-plt', type=str, default='Y')
    parser.add_argument('--h-tol', type=float, default=1e-3)
    parser.add_argument('--sigma', type=float, default=0.9)
    parser.add_argument('--n-prob', type=int, default=8 )
    parser.add_argument('--epoch', type=int, default=1000 )
    parser.add_argument('--seed-idx', type=int, default=0)
    parser.add_argument('--seed-num', type=int, default=3)
    args = parser.parse_args()
    np.random.seed(args.seed_idx)

    print('Running {} on {}'.format(args.solver_name, args.problem_name) )
    np.random.seed(args.seed_idx)
    problem = get_problem(problem_name=args.problem_name, n_var=10)
    if problem.n_obj == 2:
        args.n_prob = 8
    elif problem.n_obj == 3:
        args.n_prob = 15

    prefs = uniform_pref(n_prob=args.n_prob, n_obj = problem.n_obj, clip_eps=1e-2)
    # Actually a bit waste to implement so many solvers. Just import Core solvers.
    if args.solver_name == 'epo':
        core_solver = EPOCore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name == 'mgdaub':
        core_solver = MGDAUBCore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name == 'random':
        core_solver = RandomCore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name.startswith('agg'):
        core_solver = AggCore(n_var=problem.n_var, prefs=prefs, solver_name=args.solver_name)
    elif args.solver_name.startswith('moosvgd'):
        core_solver = MOOSVGDCore(n_var=problem.n_var, prefs=prefs)
    else:
        assert False, 'Unknown solver'

    solver = GradBaseSolver(step_size=args.step_size, epoch=args.epoch, tol=args.tol, core_solver=core_solver)

    res = solver.solve(problem=problem, x=synthetic_init(problem, prefs), prefs=prefs )
    res['prefs'] = prefs

    folder_name = os.path.join(root_name, 'Output', 'discrete', args.problem_name, args.solver_name, 'seed_{}'.format(args.seed_idx))
    os.makedirs(folder_name, exist_ok=True)

    if problem.n_obj == 2:
        plot_figure_2d(problem=problem)
    elif problem.n_obj == 3:
        plot_figure_2d()

    save_figures(folder_name=folder_name)
    save_pickles(folder_name=folder_name)




