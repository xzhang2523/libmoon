'''
    This file define the MO-Mnist problem as first proposed in Pareto multitask learning in Section 6.1.
'''
import matplotlib.pyplot as plt
import torch


from libmoon.util_global.constant import root_name
from libmoon.problem.mtl.loaders.multimnist_loader import MultiMNISTData
from libmoon.problem.mtl.objectives import CrossEntropyLoss
from libmoon.problem.mtl.model.lenet import MultiLeNet
from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global.constant import FONT_SIZE

from libmoon.solver.gradient import get_core_solver
from libmoon.solver.gradient import get_grads_from_model, numel_params


from libmoon.util_global.constant import is_pref_based


loss_1 = CrossEntropyLoss(label_name='labels_l', logits_name='logits_l')
loss_2 = CrossEntropyLoss(label_name='labels_r', logits_name='logits_r')




from tqdm import tqdm
import numpy as np
from numpy import array
import os

import itertools


class MultiMnistProblem:

    # How to train at the same time.
    def __init__(self, args):
        self.dataset = MultiMNISTData('mnist', 'train')
        self.args = args
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True,
                                                  num_workers=0)
        self.dataset_test = MultiMNISTData('mnist', 'test')
        self.loader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=0)

        self.lr = args.lr
        self.model_arr = [MultiLeNet([1, 36, 36]) for _ in range(self.args.n_prob)]
        num_params = numel_params(self.model_arr[0])
        print('num_params: ', num_params)
        for model in self.model_arr:
            model.to(args.device)

        self.is_pref_flag = is_pref_based(args.solver)
        if self.is_pref_flag:
            self.core_solver_arr = [get_core_solver(args, pref) for pref in prefs]
            self.optimizer_arr = [torch.optim.Adam(self.model_arr[idx].parameters(), lr=self.lr) for idx in
                                  range(self.n_prob)]
        else:
            self.set_core_solver = get_core_solver(args)
            params = [model.parameters() for model in self.model_arr]
            self.set_optimizer = torch.optim.Adam(itertools.chain(*params), lr=0.01)


    def optimize(self, prefs=[]):

        self.prefs = prefs
        self.n_prob = len(prefs)

        loss_all = []
        for _ in tqdm(range(self.args.num_epoch)):
            if self.is_pref_flag:
                loss_hostory = [ [] for i in range(self.n_prob) ]
            else:
                loss_hostory = []

            for data in self.loader:
                data_ = {k: v.to(self.args.device) for k, v in data.items()}
                # pref based mtd
                if self.is_pref_flag:
                    for pref_idx, (pref, model, optimizer) in enumerate(
                            zip(self.prefs, self.model_arr, self.optimizer_arr)):

                        logits_dict = self.model_arr[pref_idx](data_)
                        logits_dict['labels_l'] = data_['labels_l']
                        logits_dict['labels_r'] = data_['labels_r']

                        l1 = loss_1(**logits_dict)
                        l2 = loss_2(**logits_dict)

                        l_contains_grad = [l1, l2]

                        # here for different methods, the needed information is not enough.
                        if args.solver == 'agg':
                            pass
                        else:
                            G = get_grads_from_model(l_contains_grad, model)
                            l1_np = np.array(l1.cpu().detach().numpy(), copy=True)
                            l2_np = np.array(l2.cpu().detach().numpy(), copy=True)
                            losses = array([l1_np, l2_np])

                        if args.solver == 'agg':
                            if args.agg_mtd == 'ls':
                                alpha = self.core_solver_arr[pref_idx].get_alpha(G = None, losses=None)
                            else:
                                assert False, 'mtd not implemented'

                        else:
                            alpha = self.core_solver_arr[pref_idx].get_alpha(G = G, losses=losses)

                        self.optimizer_arr[pref_idx].zero_grad()
                        (alpha[0] * l1 + alpha[1] * l2).backward()
                        self.optimizer_arr[pref_idx].step()
                        l1_np = np.array(l1.cpu().detach().numpy(), copy=True)
                        l2_np = np.array(l2.cpu().detach().numpy(), copy=True)
                        loss_hostory[pref_idx].append([l1_np, l2_np])

                else:
                    # set based method is more complicated.
                    losses = [0,] * self.n_prob
                    losses_ts = [0] * self.n_prob

                    for model_idx, model in enumerate(self.model_arr):
                        logits_dict = self.model_arr[model_idx](data_)
                        logits_dict['labels_l'] = data_['labels_l']
                        logits_dict['labels_r'] = data_['labels_r']
                        l1 = loss_1(**logits_dict)
                        l2 = loss_2(**logits_dict)

                        losses_ts[model_idx] = torch.stack([l1, l2])
                        l1_np, l2_np = np.array(l1.cpu().detach().numpy(), copy=True), np.array(l2.cpu().detach().numpy(), copy=True)
                        losses[model_idx] = [l1_np, l2_np]

                    losses_ts = torch.stack(losses_ts)
                    losses = np.array(losses)
                    alpha = self.set_core_solver.get_alpha(losses).to(self.args.device)
                    self.set_optimizer.zero_grad()
                    torch.sum(alpha * losses_ts).backward()
                    self.set_optimizer.step()
                    loss_hostory.append(losses)

            loss_hostory = np.array(loss_hostory)
            if args.is_pref_based:
                loss_history_mean = np.mean(loss_hostory, axis=1)
            else:
                loss_history_mean = np.mean(loss_hostory, axis=0)
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
    parser.add_argument('--num_epoch', default=1, type=int)
    parser.add_argument('--use-cuda', default=True, type=bool)
    parser.add_argument('--agg-mtd', default='ls', type=str)   # This att is only valid when args.solver=agg.
    parser.add_argument('--solver', default='agg', type=str)
    parser.add_argument('--n-obj', default=2, type=int)   # This att is only valid when args.solver=agg.
    parser.add_argument('--debug', default=True, type=bool)   # This att is only valid when args.solver=agg.

    args = parser.parse_args()
    args.is_pref_based = is_pref_based(args.solver)
    if torch.cuda.is_available() and args.use_cuda:
        args.device = torch.device("cuda")  # Use the GPU
        print('cuda is available')
    else:
        args.device = torch.device("cpu")  # Use the CPU
        print('cuda is not available')


    prefs = uniform_pref(n_partition=10, n_obj=2, clip_eps=0.1)
    args.n_prob = len(prefs)
    problem = MultiMnistProblem(args, prefs)

    loss_history = problem.optimize()
    loss_history = np.array(loss_history)
    final_solution = loss_history[-1,:,:]

    for idx in range(loss_history.shape[0]):
        if idx==0:
            plt.plot(loss_history[:,idx,0], loss_history[:,idx,1], 'o-', label='pref {}'.format(idx))
        else:
            plt.plot(loss_history[:,idx,0], loss_history[:,idx,1], 'o-')



    plt.plot(final_solution[:,0], final_solution[:,1], color='k', linewidth=3)
    plt.legend(fontsize=FONT_SIZE)
    # draw pref
    solution_norm = np.linalg.norm(final_solution, axis=1, keepdims=True)
    prefs_norm = prefs / np.linalg.norm(prefs, axis=1, keepdims=True) * np.max(solution_norm)

    if args.is_pref_based:
        for pref in prefs_norm:
            plt.plot([0, pref[0]], [0, pref[1]], color='k')


    plt.xlabel('$L_1$', fontsize=FONT_SIZE)
    plt.ylabel('$L_2$', fontsize=FONT_SIZE)

    folder_name = os.path.join( root_name, 'output', args.problem, args.solver)
    os.makedirs(folder_name, exist_ok=True)
    fig_name = os.path.join(folder_name, 'final_solution.svg')
    plt.savefig(fig_name)


    plt.title('{}_{}'.format(args.problem, args.solver), fontsize= FONT_SIZE )

    print('saved in ', fig_name)

    plt.show()
