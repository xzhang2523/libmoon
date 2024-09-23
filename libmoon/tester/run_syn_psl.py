import sys
sys.path.append('D:\\pycharm_project\\libmoon')
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from libmoon.model.simple import SimplePSLModel
from torch.autograd import Variable
from libmoon.util import get_problem
from libmoon.util.constant import get_problem, FONT_SIZE, get_agg_func
from libmoon.solver.psl.core_psl import BasePSLSolver
import argparse
import os
from libmoon.util import get_uniform_pref
from torch import Tensor
from matplotlib import gridspec


def plot_psl_figure_3d(folder_name, eval_y_np):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(eval_y_np[:, 0], eval_y_np[:, 1], eval_y_np[:, 2], color='orange', s=25)
    ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
    ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
    ax.set_zlabel('$f_3$', fontsize=FONT_SIZE)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)
    ax.view_init(elev=45, azim=30)
    fig_svg_name = os.path.join(folder_name, '{}.svg'.format('res'))
    plt.savefig(fig_svg_name, bbox_inches='tight', format='svg', pad_inches=0.2)
    fig_pdf_name = os.path.join(folder_name, '{}.pdf'.format('res'))
    plt.savefig(fig_pdf_name, bbox_inches='tight', pad_inches=0.4)


def plot_psl_figure_2d(folder_name, eval_y_np, prefs, draw_fig, draw_pf, problem, labels):

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

        if problem.problem_name == 'moogaussian':
            prefs_scale = prefs * 0.1
        else:
            prefs_scale = prefs * 0.4
        for pref, y in zip(prefs_scale, eval_y_np):
            plt.scatter(pref[0], pref[1], color='blue', s=40)
            plt.scatter(y[0], y[1], color='orange', s=40)
            plt.plot([pref[0], y[0]], [pref[1], y[1]], color='tomato', linewidth=0.5, linestyle='--')
        plt.plot(prefs_scale[:, 0], prefs_scale[:, 1], color='blue', linewidth=1, label='Preference')
        plt.plot(eval_y_np[:, 0], eval_y_np[:, 1], color='orange', linewidth=1, label='Objectives')
        plt.axis('equal')
        pf = problem._get_pf(n_points=1000)
        if draw_pf == 'True':
            plt.plot(pf[:, 0], pf[:, 1], color='red', linewidth=2, label='True PF')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if labels == 'L':
            plt.xlabel('$L_1$', fontsize=FONT_SIZE)
            plt.ylabel('$L_2$', fontsize=FONT_SIZE)
        else:
            plt.xlabel('$f_1$', fontsize=FONT_SIZE)
            plt.ylabel('$f_2$', fontsize=FONT_SIZE)

        plt.axis('equal')
        plt.tight_layout()

        plt.legend(fontsize=FONT_SIZE-5, loc='upper right')
        fig_name = os.path.join( folder_name, '{}.pdf'.format('res') )
        plt.savefig(fig_name, bbox_inches='tight')
        fig_name_svg = os.path.join(folder_name, '{}.svg'.format('res'))
        plt.savefig(fig_name_svg, bbox_inches='tight')
        print('Figure saved to {}'.format(fig_name))
        print('Figure saved to {}'.format(fig_name_svg))
        if draw_fig == 'True':
            plt.show()

    else:
        if problem.problem_name == 'moogaussian':
            prefs_scale = prefs * 0.1
        else:
            prefs_scale = prefs * 0.4
        for pref, y in zip(prefs_scale, eval_y_np):
            plt.scatter(pref[0], pref[1], color='blue', s=40)
            plt.scatter(y[0], y[1], color='orange', s=40)
            plt.plot([pref[0], y[0]], [pref[1], y[1]], color='tomato', linewidth=0.5, linestyle='--')
        plt.plot(prefs_scale[:, 0], prefs_scale[:, 1], color='blue', linewidth=1, label='Preference')
        plt.plot(eval_y_np[:, 0], eval_y_np[:, 1], color='orange', linewidth=1, label='Objectives')
        plt.axis('equal')
        pf = problem._get_pf(n_points=1000)
        if draw_pf == 'True':
            plt.plot(pf[:, 0], pf[:, 1], color='red', linewidth=2, label='True PF')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if labels == 'L':
            plt.xlabel('$L_1$', fontsize=FONT_SIZE)
            plt.ylabel('$L_2$', fontsize=FONT_SIZE)
        else:
            plt.xlabel('$f_1$', fontsize=FONT_SIZE)
            plt.ylabel('$f_2$', fontsize=FONT_SIZE)

        plt.legend(fontsize=FONT_SIZE, loc='upper right')

        plt.axis('equal')
        plt.tight_layout()

        fig_name = os.path.join( folder_name, '{}.pdf'.format('res') )

        plt.savefig(fig_name, bbox_inches='tight')
        fig_name_svg = os.path.join(folder_name, '{}.svg'.format('res'))
        plt.savefig(fig_name_svg, bbox_inches='tight')
        print('Figure saved to {}'.format(fig_name))
        print('Figure saved to {}'.format(fig_name_svg))
        if draw_fig == 'True':
            plt.show()

def plot_psl_loss(folder_name, args):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Iteration', fontsize=FONT_SIZE)
    plt.ylabel('Loss', fontsize=FONT_SIZE)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    psl_loss_name = os.path.join(folder_name, '{}_loss.svg'.format(args.solver_name))
    plt.savefig(psl_loss_name, bbox_inches='tight')
    print('saved in {}'.format(psl_loss_name))
    if args.draw_fig == 'True':
        plt.show()

def save_pickle(folder_name, res):
    import pickle
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver-name', type=str, default='agg_mtche')
    parser.add_argument('--labels', type=str, default='L')
    parser.add_argument('--problem-name', type=str, default='moogaussian')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--draw-fig', type=str, default='True')
    parser.add_argument('--draw-pf', type=str, default='True')
    parser.add_argument('--eval-num', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--seed-idx', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    np.random.seed(args.seed_idx)

    problem = get_problem(problem_name=args.problem_name, n_var=10)
    args.eval_num = 20 if problem.n_obj == 2 else 200
    solver = BasePSLSolver(problem, batch_size=128, device='cuda', lr=args.lr, epoch=args.epoch,
                           solver_name=args.solver_name, use_es=False)
    print('Synthetic PSL')
    print('Running {} on {} with seed {}'.format(args.solver_name, args.problem_name, args.seed_idx))
    model, loss_history = solver.solve()

    prefs = get_uniform_pref(n_prob=args.eval_num, n_obj=problem.n_obj, clip_eps=1e-2)
    eval_y = problem.evaluate(model( Tensor(prefs).cuda())).cpu().detach().numpy()
    res = {}

    res['y'] = eval_y
    res['prefs'] = prefs
    folder_name = os.path.join('D:\\pycharm_project\\libmoon\\output\\psl', args.problem_name, args.solver_name,
                               'seed_{}'.format(args.seed_idx) )
    os.makedirs(folder_name, exist_ok=True)

    if problem.n_obj==2:
        plot_psl_figure_2d(folder_name, eval_y, prefs, args.draw_fig, args.draw_pf, problem, args.labels)
    elif problem.n_obj==3:
        plot_psl_figure_3d(folder_name, eval_y_np=eval_y)
    else:
        print('Objective {} is not drawable'.format(problem.n_obj) )

    plot_psl_loss(folder_name=folder_name, args=args)
    save_pickle(folder_name=folder_name, res=res)



