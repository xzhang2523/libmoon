import sys
sys.path.append('D:\\pycharm_project\\libmoon')
import torch
import argparse
from matplotlib import pyplot as plt
import pickle
from libmoon.solver.gradient.methods.core.core_mtl import GradBasePSLMTLSolver
import os
from libmoon.util.constant import root_name, plt_2d_tickle_size, plt_2d_marker_size, plt_2d_label_size

def save_pickle(folder_name, res):
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))


def plot_fig_2d(folder_name, eval_res, draw_fig):
    plt.figure()
    eval_loss = eval_res['eval_loss']
    plt.scatter(eval_loss[:, 0], eval_loss[:, 1])
    plt.plot(eval_loss[:, 0], eval_loss[:, 1], marker='o', markersize=plt_2d_marker_size)
    plt.xlabel('$L_1$', fontsize=plt_2d_label_size)
    plt.ylabel('$L_2$', fontsize=plt_2d_label_size)
    plt.xticks(fontsize=plt_2d_tickle_size)
    plt.yticks(fontsize=plt_2d_tickle_size)
    fig_name = os.path.join(folder_name, 'res.pdf')
    plt.savefig(fig_name, bbox_inches='tight')
    print('Save fig to {}'.format(fig_name))


def plot_train_process(folder_name, loss_history, draw_fig):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch', fontsize=plt_2d_label_size)
    plt.ylabel('Loss', fontsize=plt_2d_label_size)
    plt.grid()
    plt.xticks(fontsize=plt_2d_tickle_size)
    plt.yticks(fontsize=plt_2d_tickle_size)
    fig_name = os.path.join(folder_name, 'train_loss.pdf')
    plt.savefig(fig_name, bbox_inches='tight')
    print('Save training loss fig to {}'.format(fig_name))



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=20)
    parse.add_argument('--n-eval', type=int, default=20)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--seed-idx', type=int, default=0)
    parse.add_argument('--step-size', type=int, default=1e-4)
    parse.add_argument('--n-obj', type=int, default=2)
    parse.add_argument('--solver-name', type=str, default='epo')
    # problem_name: ['mnist', 'fashion', 'fmnist', 'adult', 'credit', 'compass']
    parse.add_argument('--problem-name', type=str, default='mnist')
    parse.add_argument('--device', type=str, default='gpu')
    parse.add_argument('--draw-fig', type=str, default='True')

    args = parse.parse_args()
    print('Device:{}'.format(args.device))
    print('Running MTL PSL {} on {} with seed {}'.format(args.solver_name, args.problem_name, args.seed_idx))
    device = torch.device('cuda') if args.device == 'gpu' else torch.device('cpu')
    solver = GradBasePSLMTLSolver(problem_name=args.problem_name, batch_size=args.batch_size,
                                  step_size=args.step_size, epoch=args.epoch, device=device, solver_name=args.solver_name)

    # Training Loss
    print('Training...')
    train_res = solver.solve()
    loss_history = train_res['train_loss']
    print('Evaluating...')
    eval_res = solver.eval(n_eval=args.n_eval)
    eval_res['y'] = eval_res['eval_loss']

    folder_name = os.path.join(root_name, 'Output', 'psl', args.problem_name, args.solver_name,
                               'seed_{}'.format(args.seed_idx))
    os.makedirs(folder_name, exist_ok=True)
    plot_train_process(folder_name=folder_name, loss_history=loss_history, draw_fig=args.draw_fig)
    plot_fig_2d(folder_name=folder_name, eval_res=eval_res, draw_fig=args.draw_fig)
    save_pickle(folder_name=folder_name, res=eval_res)