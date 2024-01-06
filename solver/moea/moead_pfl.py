from solver.moea.moead import MOEAD
import numpy as np
import argparse
import time
from pymoo.indicators.hv import HV
from util_global.constant import FONT_SIZE, problem_dict

from solver.moea.utils.population import population, external_population
from solver.moea.utils.termination import termination
from solver.moea.utils.genetic_operator import cross_sbx, mut_pm
from solver.moea.utils.utils_ea import population_initialization, neighborhood_selection, OperatorDE

from matplotlib import pyplot as plt
from solver.pfl.model.simple import PFLModel
import torch
from torch import Tensor
from torch.optim import Adam

from torch.autograd import Variable


class MOEAD_PFL(MOEAD):
    def __init__(self,
                 mop: any = None,
                 ref_vec: np.ndarray = None,
                 n_neighbors: int = 10,
                 ):
        super().__init__(mop, ref_vec, n_neighbors)
        self.pfl_update_num = 10
        self.name = 'MOEA/D-PFL'
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




    def pfl_update(self):
        loss_arr = []
        for idx in range(500):
            pred_y = self.pfl_model.forward( Tensor(self.ref_vec) )
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
        pref_ts = Variable( Tensor(pref), requires_grad=True )
        pref_optimizer = Adam([pref_ts], lr=0.001)

        # print()




    def step(self):
        # print('using PFL updation')
        self.gen += 1
        if self.gen % self.pfl_update_num == 0:
            self.pfl_update()
            self.pref_adjust()

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
    parser.add_argument('--n-gen', type=int, default=1000)
    parser.add_argument('--crossover', type=str, default='sbx')  # crossover operator ['de', 'sbx']
    parser.add_argument('--problem-name', type=str, default='RE21')  # should be in lowwer case

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

    if problem.n_obj == 2:
        plt.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], c='none', edgecolors='r')
        plt.plot(alg.pop.F[:, 0], alg.pop.F[:, 1], c='r')

        if not problem.problem_name.startswith('RE'):
            pf = problem.get_pf()
            plt.plot(pf[:, 0], pf[:, 1], c='b')

        plt.xlabel('$f_1$', fontsize=FONT_SIZE )
        plt.ylabel('$f_2$', fontsize=FONT_SIZE )

        ref_vec_norm = ref_vec / np.linalg.norm(ref_vec, axis=1, keepdims=True)
        for ref in ref_vec_norm:
            plt.plot([0, ref[0]], [0, ref[1]], c='k')

    plt.show()

