import matplotlib.pyplot as plt
from util.constant import root_name
from problem.mtl.loaders.multimnist_loader import MultiMNISTData
import torch

from problem.mtl.objectives import CrossEntropyLoss
from problem.mtl.model.simple import MultiLeNet
from solver.gradient.core_solver import CoreAgg
from util.weight_factor.funs import uniform_pref
from util.constant import FONT_SIZE

loss_1 = CrossEntropyLoss(label_name='labels_l', logits_name='logits_l')
loss_2 = CrossEntropyLoss(label_name='labels_r', logits_name='logits_r')
from tqdm import tqdm
import numpy as np
from numpy import array
import os
from solver.gradient import get_core_solver
from solver.gradient.util import get_grads_from_model, numel_params

from util.constant import is_pref_based


class MultiMnistProblem:

    # How to train at the same time.
    def __init__(self, args, prefs):
        self.dataset = MultiMNISTData('mnist', 'train')
        self.args = args
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True,
                                                  num_workers=0)
        self.dataset_test = MultiMNISTData('mnist', 'test')
        self.loader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=0)
        self.lr = args.lr
        self.prefs = prefs
        self.n_prob = len(prefs)

        self.model_arr = [MultiLeNet([1, 36, 36]) for _ in range(self.n_prob)]

        num_params = numel_params(self.model_arr[0])
        print('num_params: ', num_params)

        for model in self.model_arr:
            model.to(args.device)

        self.optimizer_arr = [torch.optim.Adam(self.model_arr[idx].parameters(), lr=self.lr) for idx in
                              range(self.n_prob)]

        if is_pref_based(args.mtd):
            self.core_solver_arr = [get_core_solver(args.mtd, args.agg_mtd, pref) for pref in prefs]
        else:
            self.set_core_solver = get_core_solver(args.mtd)



    def optimize(self):
        loss_all = []
        for _ in tqdm(range(self.args.num_epoch)):
            loss_hostory = [[] for i in range(self.n_prob)]
            for data in self.loader:
                data_ = {k: v.to(self.args.device) for k, v in data.items()}
                for pref_idx, (pref, model, optimizer) in enumerate(
                        zip(self.prefs, self.model_arr, self.optimizer_arr)):
                    logits_dict = self.model_arr[pref_idx](data_)
                    logits_dict['labels_l'] = data_['labels_l']
                    logits_dict['labels_r'] = data_['labels_r']
                    l1 = loss_1(**logits_dict)
                    l2 = loss_2(**logits_dict)

                    l_contains_grad = [l1, l2]
                    G = get_grads_from_model(l_contains_grad, model)

                    l1_np = np.array(l1.cpu().detach().numpy(), copy=True)
                    l2_np = np.array(l2.cpu().detach().numpy(), copy=True)
                    losses = array([l1_np, l2_np])
                    alpha = self.core_solver_arr[pref_idx].get_alpha(G = G, losses=losses)
                    self.optimizer_arr[pref_idx].zero_grad()
                    (alpha[0] * l1 + alpha[1] * l2).backward()
                    self.optimizer_arr[pref_idx].step()
                    l1_np = np.array(l1.cpu().detach().numpy(), copy=True)
                    l2_np = np.array(l2.cpu().detach().numpy(), copy=True)
                    loss_hostory[pref_idx].append([l1_np, l2_np])
            loss_hostory = np.array(loss_hostory)
            loss_history_mean = np.mean(loss_hostory, axis=1)
            loss_all.append(loss_history_mean)
        return loss_all




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='mnist', type=str)  # For attribute in args, we all call problem.
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--use-cuda', default=True, type=bool)

    parser.add_argument('--mtd', default='epo', type=str)
    parser.add_argument('--agg-mtd', default='ls', type=str)   # This att is only valid when args.mtd=agg.

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda")  # Use the GPU
        print('cuda is available')
    else:
        device = torch.device("cpu")  # Use the CPU
        print('cuda is not available')

    args.device = device
    prefs = uniform_pref(n_partition=3, n_obj=2, clip_eps=0.1)

    problem = MultiMnistProblem(args, prefs)
    loss_history = problem.optimize()
    loss_history = np.array(loss_history)

    final_solution = loss_history[-1,:,:]
    plt.scatter(final_solution[:,0], final_solution[:,1], label='final solution')
    plt.legend(fontsize=FONT_SIZE)
    # draw pref

    solution_norm = np.linalg.norm(final_solution, axis=1, keepdims=True)
    prefs_norm = prefs / np.linalg.norm(prefs, axis=1, keepdims=True) * solution_norm
    for pref in prefs_norm:
        plt.plot([0, pref[0]], [0, pref[1]], color='k')

    plt.xlabel('$L_1$', fontsize=FONT_SIZE)
    plt.ylabel('$L_2$', fontsize=FONT_SIZE)


    folder_name = os.path.join( root_name, 'output', args.problem, args.mtd)
    os.makedirs(folder_name, exist_ok=True)
    fig_name = os.path.join(folder_name, 'final_solution.svg')
    plt.savefig(fig_name)
    print('saved in ', fig_name)

    plt.show()
