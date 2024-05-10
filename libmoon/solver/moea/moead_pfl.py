import os.path

from solver.moea.moead import MOEAD
import numpy as np
import argparse
import time
from pymoo.indicators.hv import HV
from libmoon.util_global.constant import FONT_SIZE, problem_dict, root_name

from solver.moea.utils.genetic_operator import cross_sbx, mut_pm
from solver.moea.utils.utils_ea import neighborhood_selection, OperatorDE

from matplotlib import pyplot as plt
from solver.pfl.model.simple import PFLModel
import torch
from torch import Tensor
from torch.optim import Adam

from torch.autograd import Variable
from libmoon.metrics.metrics import get_MMS, pref2angle, angle2pref

import pickle



class MOEAD_PFL(MOEAD):
    def __init__(self,
                 mop: any = None,
                 ref_vec: np.ndarray = None,
                 n_neighbors: int = 10,
                 ):
        super().__init__(mop, ref_vec, n_neighbors)
        # self.pfl_update_num = 100
        self.name = 'MOEAD_PFL'
        # self.pfl_model = PFLModel(self.mop.n_obj)

    def setup(self,
              mop: any = None,
              args: argparse.Namespace = None,
              max_gen: int = 100,
              pop_size: int = 100,
              ) -> None:

        super().setup(mop, args, max_gen, pop_size)
        self.pfl_model = PFLModel(self.mop.n_obj)

        self.optimizer = Adam(self.pfl_model.parameters(), lr=0.001)
        self.pfl_update_num = args.n_pfl_update


    def pfl_update(self):
        loss_arr = []

        angle_ts = pref2angle(self.ref_vec, self.mop.n_obj)
        angle_ts = torch.Tensor( angle_ts )

        for idx in range(500):
            pred_y = self.pfl_model.forward( angle_ts )
            loss = torch.sum( torch.pow(pred_y - Tensor(self.pop.F), 2) )
            # print()
            self.optimizer.zero_grad()
            loss.backward()
            loss_arr.append(loss.item())
            self.optimizer.step()


        use_plot = False
        if use_plot:
            plt.plot(loss_arr)
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.show()
            assert False


    def pref_adjust(self):
        pref = self.ref_vec
        pref_old = np.copy(pref)
        angle = pref2angle(pref, self.mop.n_obj)
        angle_ts = Variable( Tensor(angle), requires_grad=True )
        angle_optimizer = Adam([angle_ts], lr=0.01)

        loss_arr = []
        for idx in range(1000):
            pred_y = self.pfl_model.forward(angle_ts)
            mms = get_MMS( pred_y )
            angle_optimizer.zero_grad()
            mms.backward()
            loss_arr.append(mms.item())

            angle_optimizer.step()
            angle_ts.data = torch.clip(angle_ts.data, 1e-3, np.pi/2-1e-3)

        pref_ts = angle2pref(angle_ts, self.mop.n_obj)
        self.ref_vec = pref_ts.detach().numpy()


        use_plot = False
        if use_plot:
            plt.plot(loss_arr)
            plt.xlabel('iteration')
            plt.ylabel('mms')
            plt.show()
            assert False

        return pref_old




    def step(self):
        # print('using PFL updation')
        self.gen += 1

        if self.gen % self.pfl_update_num == 0:

            # for idx, ref in enumerate(self.ref_vec):
            #     if idx == 0:
            #         plt.scatter(ref[0], ref[1], label='old', color='r')
            #     else:
            #         plt.scatter(ref[0], ref[1], color='r')

            self.pfl_update()
            pref_old = self.pref_adjust()
            print('adjust at {}'.format(self.gen))

            use_plt=False
            if use_plt:
                plt.scatter(pref_old[:,0], pref_old[:,1], label='old', color='r')
                plt.scatter(self.ref_vec[:,0], self.ref_vec[:,1], label='new', color='b')
                plt.legend()
                plt.show()

        for k in np.random.permutation(self.n_pop):
            if self.args.crossover == 'sbx':
                P = neighborhood_selection(self.n_pop, self.neighbors[k])
                X = self.pop.parent_select(P)
                off_cross = cross_sbx(X, self.mop.get_lower_bound, self.mop.get_upper_bound)
            else:
                P = neighborhood_selection(self.n_pop, self.neighbors[k], n_selects=3)
                X = self.pop.parent_select(P)
                off_cross = OperatorDE(np.atleast_2d(X[0]), np.atleast_2d(X[1]), np.atleast_2d(X[2]), self.mop)
            off = mut_pm(off_cross, self.mop.get_lower_bound, self.mop.get_upper_bound)
            off_f = self.mop(off).squeeze()
            self.z_star = np.minimum(self.z_star, off_f)
            self._update_neighbor(off, off_f, self.neighbors[k])
            self.ep(off, off_f)
        self.termina(nfe=self.n_pop, gen=1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-gen', type=int, default=2000)
    parser.add_argument('--n-pfl-update', type=int, default=500)

    parser.add_argument('--crossover', type=str, default='sbx')  # crossover operator ['de', 'sbx']
    parser.add_argument('--problem-name', type=str, default='RE24')  # should be in lowwer case

    args = parser.parse_args()
    problem = problem_dict[args.problem_name]
    ref_point = np.array(1.2 * np.ones(problem.n_obj))
    ind = HV(ref_point=ref_point)

    alg = MOEAD_PFL()
    print('{} on {}'.format(alg.name, problem.problem_name))
    alg.setup(problem, args, max_gen=args.n_gen, pop_size=10)

    ref_vec = alg.ref_vec
    ts = time.time()
    alg.solve()
    print('solving over {:.2f}m'.format((time.time() - ts) / 60))

    # statistics and save.
    hv_val = ind.do(alg.pop.F)
    mms = get_MMS(alg.pop.F)

    print('hv: {:.4f}'.format(hv_val))
    print('mms: {:.4f}'.format(mms))


    pickle_folder = os.path.join(root_name, 'output', problem.problem_name, alg.name)
    os.makedirs(pickle_folder, exist_ok=True)
    pickle_name = os.path.join(pickle_folder, 'res.pkl')
    with open(pickle_name, 'wb') as f:
        pickle.dump(alg.pop.F, f)

    txt_name = os.path.join(pickle_folder, 'res.txt')
    with open(txt_name, 'w') as f:
        f.write('hv: {:.4f}\n'.format(hv_val))
        f.write('mms: {:.4f}\n'.format(mms))


    if problem.n_obj == 2:
        plt.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], c='none', edgecolors='r', label='solution')
        plt.plot(alg.pop.F[:, 0], alg.pop.F[:, 1], c='r')

        if not problem.problem_name.startswith('RE'):
            pf = problem.get_pf()
            plt.plot(pf[:, 0], pf[:, 1], c='b')

        plt.legend()
        plt.xlabel('$f_1$', fontsize=FONT_SIZE )
        plt.ylabel('$f_2$', fontsize=FONT_SIZE )

        ref_vec_norm = ref_vec / np.linalg.norm(ref_vec, axis=1, keepdims=True)
        for ref in ref_vec_norm:
            plt.plot([0, ref[0]], [0, ref[1]], c='k')
    elif problem.n_obj == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], alg.pop.F[:, 2], c='none', edgecolors='r')
        ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
        ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
        ax.set_zlabel('$f_3$', fontsize=FONT_SIZE)
    else:
        assert False, 'n_obj should be 2 or 3'

    plt.show()

