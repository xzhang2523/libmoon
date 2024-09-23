import argparse
import numpy as np
import sys
sys.path.append('D:\\pycharm_project\\libmoon\\')
from libmoon.util import synthetic_init, get_problem, get_uniform_pref
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.solver.gradient.methods.core.core_solver import EPOCore, MGDAUBCore, RandomCore, AggCore, MOOSVGDCore, HVGradCore, PMTLCore
from libmoon.solver.gradient.methods.core.core_solver import PMGDACore
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import gridspec
from matplotlib import pyplot as plt
from libmoon.util.constant import FONT_SIZE_2D, FONT_SIZE_3D, color_arr, beautiful_dict, root_name, min_key_array
from libmoon.util.constant import plt_2d_tickle_size, plt_2d_label_size
def draw_2d_prefs(prefs, rho):
    prefs_norm2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    for idx, pref in enumerate(prefs_norm2):
        plt.plot([0, pref[0]*rho], [0, pref[1]*rho], color='grey', linewidth=2,
                 linestyle='--')

def plot_figure_2d(problem, labels):
    if problem.problem_name == 'moogaussian':
        gs = gridspec.GridSpec(3, 3)
        ax1 = plt.subplot(gs[0, 1:])  # Span across the top row
        ax1.set_xlim([-10, 10])  # Set x-axis limits explicitly
        x = np.linspace(-10, 10, 500)
        ax1.plot(x, np.exp(-x ** 2), 'r')

        # Second subplot (left column, covering rows 1 and 2)
        ax2 = plt.subplot(gs[1:, 0])  # Span across rows 1 and 2
        x = np.linspace(-10, 10, 500)
        ax2.plot(np.exp(-(x - 1) ** 2), x, 'r')

        ax2.tick_params(axis='x', labelrotation=45)  # 旋转x轴刻度标签45度
        ax2.tick_params(axis='y', labelrotation=90)  # 旋转y轴刻度标签90度

        # plt.subplot(gs[1:, 1:], sharex=ax1, sharey=ax2)
        plt.subplot(gs[1:, 1:])

        # pl.imshow(M, interpolation='nearest')
        y_arr = res['y']
        rho = np.max([np.linalg.norm(y) for y in y_arr])

        plt.scatter(y_arr[:, 0], y_arr[:, 1], color='black')
        plt.xticks(fontsize=plt_2d_tickle_size)
        plt.yticks(fontsize=plt_2d_tickle_size)
        if labels == 'L':
            plt.xlabel('$L_1$', fontsize=plt_2d_label_size)
            plt.ylabel('$L_2$', fontsize=plt_2d_label_size)
        else:
            plt.xlabel('$f_1$', fontsize=plt_2d_label_size)
            plt.ylabel('$f_2$', fontsize=plt_2d_label_size)

        # plt.axis('equal')
        if hasattr(problem, '_get_pf'):
            pf = problem._get_pf(n_points=1000)
            plt.plot(pf[:, 0], pf[:, 1], color='red', linewidth=2, label='True PF')
            plt.legend(fontsize=15)
        draw_2d_prefs(prefs, rho)

        plt.axis('equal')
        plt.tight_layout()

    else:
        y_arr = res['y']
        rho = np.max([np.linalg.norm(y) for y in y_arr])

        plt.scatter(y_arr[:, 0], y_arr[:, 1], color='black')
        plt.xticks(fontsize=plt_2d_tickle_size)
        plt.yticks(fontsize=plt_2d_tickle_size)
        if labels == 'L':
            plt.xlabel('$L_1$', fontsize=plt_2d_label_size)
            plt.ylabel('$L_2$', fontsize=plt_2d_label_size)
        else:
            plt.xlabel('$f_1$', fontsize=plt_2d_label_size)
            plt.ylabel('$f_2$', fontsize=plt_2d_label_size)

        plt.axis('equal')
        if hasattr(problem, '_get_pf'):
            pf = problem._get_pf(n_points=1000)
            plt.plot(pf[:, 0], pf[:, 1], color='red', linewidth=2, label='True PF')
            plt.legend(fontsize=15)
        draw_2d_prefs(prefs, rho)



def plot_figure_3d(folder_name):
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
    print('Save fig to {}'.format(fig_name_svg))
    plt.title(beautiful_dict[args.solver_name])

def save_pickles(folder_name):
    import pickle
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script')
    # mgdaub random epo pmgda agg_ls agg_tche agg_pbi agg_cosmos, agg_softtche pmtl hvgrad moosvgd
    parser.add_argument('--solver-name', type=str, default='agg_mtche')
    parser.add_argument( '--problem-name', type=str, default='moogaussian')
    parser.add_argument( '--labels', type=str, default='L')

    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--draw-fig', type=str, default='True')
    parser.add_argument('--n-prob', type=int, default=10 )
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--seed-idx', type=int, default=1)

    args = parser.parse_args()
    np.random.seed(args.seed_idx)
    print('Synthetic discrete')
    print('Running {} on {} with seed {}'.format(args.solver_name, args.problem_name, args.seed_idx) )
    np.random.seed(args.seed_idx)
    problem = get_problem(problem_name=args.problem_name, n_var=1)
    prefs = get_uniform_pref(n_prob=args.n_prob, n_obj = problem.n_obj, clip_eps=1e-2)

    # Actually a bit waste to implement so many solvers. Just import Core solvers.
    if args.solver_name == 'epo':
        core_solver = EPOCore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name == 'mgdaub':
        core_solver = MGDAUBCore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name == 'random':
        core_solver = RandomCore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name == 'pmgda':
        core_solver = PMGDACore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name.startswith('agg'):
        core_solver = AggCore(n_var=problem.n_var, prefs=prefs, solver_name=args.solver_name)
    elif args.solver_name == 'moosvgd':
        core_solver = MOOSVGDCore(n_var=problem.n_var, prefs=prefs)
    elif args.solver_name == 'hvgrad':
        core_solver = HVGradCore(n_obj=problem.n_obj, n_var=problem.n_var, problem_name=problem.problem_name)
    elif args.solver_name == 'pmtl':
        core_solver = PMTLCore(n_obj=problem.n_obj, n_var=problem.n_var, total_epoch=args.epoch, warmup_epoch=args.epoch // 5, prefs=prefs)
    else:
        assert False, 'Unknown solver'

    solver = GradBaseSolver(step_size=args.step_size, epoch=args.epoch, tol=args.tol, core_solver=core_solver)
    res = solver.solve(problem=problem, x=synthetic_init(problem, prefs), prefs=prefs )
    res['prefs'] = prefs
    folder_name = os.path.join(root_name, 'Output', 'discrete', args.problem_name, args.solver_name,
                               'seed_{}'.format(args.seed_idx))


    os.makedirs(folder_name, exist_ok=True)
    if problem.n_obj == 2:
        plot_figure_2d(problem=problem, labels=args.labels)
    elif problem.n_obj == 3:
        assert False, 'Method not implemented'


    save_figures(folder_name=folder_name)
    save_pickles(folder_name=folder_name)




