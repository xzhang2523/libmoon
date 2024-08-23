import sys
sys.path.append('D:\\pycharm_project\\libmoon')
import torch
import argparse
import numpy as np
from libmoon.problem.mtl.objectives import from_name
from matplotlib import pyplot as plt
import pickle
from libmoon.util.network import numel
from libmoon.solver.gradient.methods.core.core_mtl import GradBasePSLMTLSolver
import os
from libmoon.util.constant import root_name

from libmoon.util.mtl import get_dataset, model_from_dataset

def save_pickle(folder_name, res):
    pickle_name = os.path.join(folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))

def plot_fig_2d(folder_name):
    fig = plt.figure()
    plt.scatter(loss_ray_np[:, 0], loss_ray_np[:, 1])
    plt.xlabel('Loss 1', fontsize=18)
    plt.ylabel('Loss 2', fontsize=18)
    save_fig_name = os.path.join(folder_name, 'loss_ray.pdf')
    plt.savefig(save_fig_name)
    print('save to {}'.format(save_fig_name))

def plot_train_process(folder_name, loss_history):
    fig = plt.figure()
    plt.plot(np.array(loss_history))
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Hypernet loss', fontsize=18)
    save_fig_name = os.path.join(folder_name, 'hypernet_loss.pdf')
    plt.savefig(save_fig_name)
    print('save to {}'.format(save_fig_name))


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=10)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--step-size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--n-obj', type=int, default=2)
    parse.add_argument('--solver', type=str, default='agg_ls')

    # problem_name: ['mnist', 'fashion', 'mfashion']
    parse.add_argument('--problem-name', type=str, default='')
    parse.add_argument('--device', type=str, default='gpu')
    parse.add_argument('--n-eval', type=int, default=10)

    args = parse.parse_args()
    print('Device:{}'.format(args.device))

    model = model_from_dataset(args.problem_name)
    num_param = numel(model)
    print('Number of parameters: {}'.format(num_param))

    device = torch.device('cuda') if args.device == 'gpu' else torch.device('cpu')
    solver = GradBasePSLMTLSolver(problem_name=args.problem_name, batch_size=args.batch_size,
                                  step_size=args.step_size, epoch=args.epoch, device=device)

    train_res = solver.solve()
    eval_res = solver.eval(n_eval=args.n_eval)

    loss, prefs = eval_res['loss'], eval_res['prefs']
    res = {}

    res['y'] = loss
    res['prefs'] = prefs


    folder_name = os.path.join(root_name, 'Output', 'psl', args.problem_name, args.solver_name,
                               'seed_{}'.format(args.seed_idx))
    os.makedirs(folder_name, exist_ok=True)


    plot_fig_2d(folder_name=folder_name, loss=loss, prefs=prefs)
    save_pickle(folder_name=folder_name)







