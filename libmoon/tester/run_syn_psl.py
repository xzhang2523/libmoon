import sys
sys.path.append('D:\\pycharm_project\\libmoon')
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from libmoon.solver.psl.model import SimplePSLModel
from torch.autograd import Variable
from libmoon.util_global import get_problem
from libmoon.util_global.constant import get_problem, FONT_SIZE, get_agg_func
from libmoon.solver.psl.core_psl import BasePSLSolver
import argparse
import os
from libmoon.util_global import uniform_pref
from torch import Tensor


def plot_psl_figure_3d(solver_name, problem_name, prefs, draw_fig):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(eval_y_np[:, 0], eval_y_np[:, 1], eval_y_np[:, 2], color='orange', s=40)

    ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
    ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
    ax.set_zlabel('$f_3$', fontsize=FONT_SIZE)

    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\output\\psl', problem_name)
    os.makedirs(folder_name, exist_ok=True)
    fig_name = os.path.join(folder_name, '{}.pdf'.format(solver_name))
    plt.savefig(fig_name, bbox_inches='tight')
    fig_name_svg = os.path.join(folder_name, '{}.svg'.format(solver_name))
    plt.savefig(fig_name_svg, bbox_inches='tight')
    print('Figure saved to {}'.format(fig_name))
    print('Figure saved to {}'.format(fig_name_svg))
    if draw_fig == 'True':
        plt.show()




def plot_psl_figure_2d(solver_name, problem_name, prefs, draw_fig):
    prefs_scale = prefs * 0.4
    for pref, y in zip(prefs_scale, eval_y_np):
        plt.scatter(pref[0], pref[1], color='blue', s=40)
        plt.scatter(y[0], y[1], color='orange', s=40)
        plt.plot([pref[0], y[0]], [pref[1], y[1]], color='tomato', linewidth=0.5, linestyle='--')
    plt.plot(prefs_scale[:, 0], prefs_scale[:, 1], color='blue', linewidth=1, label='Preference')
    plt.plot(eval_y_np[:, 0], eval_y_np[:, 1], color='orange', linewidth=1, label='Objectives')
    plt.axis('equal')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('$f_1$', fontsize=FONT_SIZE)
    plt.ylabel('$f_2$', fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\output\\psl', problem_name)
    os.makedirs(folder_name, exist_ok=True)
    fig_name = os.path.join( folder_name, '{}.pdf'.format(solver_name) )
    plt.savefig(fig_name, bbox_inches='tight')
    fig_name_svg = os.path.join(folder_name, '{}.svg'.format(solver_name))
    plt.savefig(fig_name_svg, bbox_inches='tight')

    print('Figure saved to {}'.format(fig_name))
    print('Figure saved to {}'.format(fig_name_svg))

    if draw_fig == 'True':
        plt.show()

def plot_psl_loss():
    fig = plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Iteration', fontsize=FONT_SIZE)
    plt.ylabel('Loss', fontsize=FONT_SIZE)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\output\\psl', args.problem_name)
    psl_loss_name = os.path.join(folder_name, '{}_loss.pdf'.format(args.solver_name))
    plt.savefig(psl_loss_name, bbox_inches='tight')
    print('saved in {}'.format(psl_loss_name))
    if args.draw_fig == 'True':
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # solver array: ['agg_ls', 'agg_tche', 'agg_pbi', 'agg_cosmos', 'epo']
    parser.add_argument('--solver-name', type=str, default='agg_mtche')
    # problem array: ['VLMOP1', 'RE37' ]
    parser.add_argument('--problem-name', type=str, default='RE37')
    parser.add_argument('--draw-fig', type=str, default='True')
    parser.add_argument('--epoch', type=int, default=500)
    args = parser.parse_args()
    problem = get_problem(problem_name=args.problem_name, n_var=10)
    solver = BasePSLSolver(problem, batch_size=128, device='cuda', lr=1e-3, epoch=args.epoch,
                           solver_name=args.solver_name, use_es=False)
    print('Running {} on {}'.format(args.solver_name, args.problem_name))
    model, loss_history = solver.solve()

    prefs = uniform_pref(n_prob=200, n_obj=problem.n_obj, clip_eps=1e-2)
    eval_y_np = problem.evaluate(model( Tensor(prefs).cuda())).cpu().detach().numpy()


    if problem.n_obj==2:
        plot_psl_figure_2d(solver_name=args.solver_name, problem_name=args.problem_name,
                           prefs=prefs, draw_fig=args.draw_fig)
    elif problem.n_obj==3:
        plot_psl_figure_3d(solver_name=args.solver_name, problem_name=args.problem_name,
                           prefs=prefs, draw_fig=args.draw_fig)
    else:
        print('Objective {} is not drawable'.format(problem.n_obj) )


    plot_psl_loss()


