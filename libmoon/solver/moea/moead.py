import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

import copy

from pymoo.util.ref_dirs import get_reference_directions
from solver.moea.utils import get_decomposition
from solver.moea.utils.population import population, external_population
from libmoon.solver.moea.utils.termination import termination
from solver.moea.utils.genetic_operator import cross_sbx, mut_pm
from solver.moea.utils.utils_ea import population_initialization, neighborhood_selection, OperatorDE
from libmoon.util_global.constant import FONT_SIZE, problem_dict, root_name

from libmoon.visulization.util import plot_simplex, plot_unit_sphere
import argparse
import time
from pymoo.indicators.hv import HV
from libmoon.metrics.metrics import compute_indicators

import pickle
import os


class MOEAD():
    def __init__(self,
                 mop: any = None,
                 ref_vec: np.ndarray = None,
                 n_neighbors: int = 10,
                 ):

        self.name = 'MOEAD'
        self.mop = copy.deepcopy(mop)
        self.ref_vec = ref_vec
        self.n_neighbors = n_neighbors
        self.pop = population()
        self.ep = external_population()
        self.termina = termination()

    @property
    def solution_ep(self):
        return self.ep.non_dominate_sol

    @property
    def solution_pop(self):
        return self.pop.X, self.pop.F

    def setup(self,
              mop: any = None,
              args: argparse.Namespace = None,
              max_gen: int = 100,
              pop_size: int = 100,
              ) -> None:


        self.args = args
        if mop is not None:
            self.mop = mop
        assert not mop.has_constraint
        self.termina.setup(max_gen=max_gen)

        if self.ref_vec is None:
            self.ref_vec = get_reference_directions("uniform", mop.get_number_objective, n_partitions=pop_size-1)
            self.ref_vec = np.clip(self.ref_vec, 1e-4, 1-1e-4)


        self.n_pop = len(self.ref_vec)
        pop = population_initialization(self.n_pop, self.mop)
        f = self.mop(pop)
        self.z_star = np.min(f, axis=0)

        self.pop(pop, f)
        self.termina(nfe=self.n_pop)
        self.neighbors = np.argsort(cdist(self.ref_vec, self.ref_vec), axis=1)[:, :self.n_neighbors]

        self.decomposition = get_decomposition('tch')
        self.gen = 0


    def solve(self):
        while self.termina.has_next:
            self.step()
            if self.gen % 500 == 0:
                print('gen: {}'.format(self.gen))


    def reset(self,
              problem: any):
        """
            Initialization.
        """

    def step(self):
        self.gen += 1
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

    def _update_neighbor(self, off, off_f, neighbors):
        for i in neighbors:
            if self.decomposition(off_f, self.ref_vec[i], self.z_star) <= self.decomposition(self.pop.F[i],
                                                                                             self.ref_vec[i],

                                                                                             self.z_star): self.pop(off, F=off_f, ind=i)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-gen', type=int, default=2000 )
    parser.add_argument('--crossover', type=str, default='sbx')   # crossover operator ['de', 'sbx']
    parser.add_argument('--problem-name', type=str, default='ZDT4')  # should be in lowwer case
    args = parser.parse_args()

    problem = problem_dict[args.problem_name]
    ref_point = np.array( 1.2 * np.ones(problem.n_obj) )
    ind = HV(ref_point=ref_point)

    alg = MOEAD()
    print('{} on {}'.format(alg.name, problem.problem_name))
    alg.setup(problem, args, max_gen=args.n_gen, pop_size=10)

    ref_vec = alg.ref_vec
    ts = time.time()
    alg.solve()
    print( 'solving over {:.2f}m'.format( (time.time() - ts )/60))
    indicator_res = compute_indicators(alg.pop.F)

    if problem.n_obj == 2:
        plt.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], c='none', edgecolors='r', label='solution')
        plt.plot(alg.pop.F[:, 0], alg.pop.F[:, 1], c='r')

        if not problem.problem_name.startswith('RE'):
            pf = problem.get_pf()
            plt.plot(pf[:, 0], pf[:, 1], c='b', label='Pareto front')

        plt.xlabel('$f_1$', fontsize=FONT_SIZE )
        plt.ylabel('$f_2$', fontsize=FONT_SIZE )

        ref_vec_norm = ref_vec / np.linalg.norm(ref_vec, axis=1, keepdims=True)
        for ref in ref_vec_norm:
            plt.plot([0, ref[0]], [0, ref[1]], c='k')

    else:
        if problem.n_obj==3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], alg.pop.F[:, 2], c='none', edgecolors='r')
            ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
            ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
            ax.set_zlabel('$f_3$', fontsize=FONT_SIZE)

            ref_norm = np.copy(alg.ref_vec)
            ref_norm = ref_norm / np.linalg.norm(ref_norm, axis=1, keepdims=True)

            for ref in ref_norm:
                if problem.problem_name == 'DTLZ1':
                    ax.plot([0, 0.5*ref[0]], [0, 0.5*ref[1]], [0, 0.5*ref[2]], c='k')
                else:
                    ax.plot([0, ref[0]], [0, ref[1]], [0, ref[2]], c='k')

        elif problem.n_obj==4:
            # plt.subplot(4, 1, 1)
            fig = plt.figure(figsize=(4, 10))
            ax = fig.add_subplot(4, 1, 1, projection='3d')
            ax.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], alg.pop.F[:, 2], c='none', edgecolors='r')
            ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
            ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
            ax.set_zlabel('$f_3$', fontsize=FONT_SIZE)
            ax.set_title('$f_1, f_2, f_3$')

            ax = fig.add_subplot(4, 1, 2, projection='3d')
            ax.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], alg.pop.F[:, 3], c='none', edgecolors='r')
            ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
            ax.set_ylabel('$f_2$', fontsize=FONT_SIZE)
            ax.set_zlabel('$f_4$', fontsize=FONT_SIZE)
            ax.set_title('$f_1, f_2, f_4$')

            ax = fig.add_subplot(4, 1, 3, projection='3d')
            ax.scatter(alg.pop.F[:, 0], alg.pop.F[:, 2], alg.pop.F[:, 3], c='none', edgecolors='r')
            ax.set_xlabel('$f_1$', fontsize=FONT_SIZE)
            ax.set_ylabel('$f_3$', fontsize=FONT_SIZE)
            ax.set_zlabel('$f_4$', fontsize=FONT_SIZE)
            ax.set_title('$f_1, f_3, f_4$')

            ax = fig.add_subplot(4, 1, 4, projection='3d')
            ax.scatter(alg.pop.F[:, 1], alg.pop.F[:, 2], alg.pop.F[:, 3], c='none', edgecolors='r')
            ax.set_xlabel('$f_2$', fontsize=FONT_SIZE)
            ax.set_ylabel('$f_3$', fontsize=FONT_SIZE)
            ax.set_zlabel('$f_4$', fontsize=FONT_SIZE)
            ax.set_title('$f_2, f_3, f_4$')


        if problem.problem_name == 'DTLZ1':
            p1 = np.array([0, 0, 0.5])
            p2 = np.array([0, 0.5, 0.0])
            p3 = np.array([0.5, 0, 0.0])
            plot_simplex(ax, p1, p2, p3)
        elif problem.problem_name.startswith('DTLZ'):
            plot_unit_sphere(ax)

    plt.legend()



    pickle_folder = os.path.join(root_name, 'output', problem.problem_name, alg.name)
    os.makedirs(pickle_folder, exist_ok=True)
    pickle_name = os.path.join(pickle_folder, 'res.pkl')
    with open(pickle_name, 'wb') as f:
        pickle.dump(alg.pop.F, f)
        pickle.dump(indicator_res, f)


    txt_name = os.path.join(pickle_folder, 'res.txt')
    with open(txt_name, 'w') as f:
        for k, v in indicator_res.items():
            f.write('{}: {}\n'.format(k, v))

    plt.title('MOEA/D({}) on {} with {}'.format(args.crossover, problem.problem_name, args.n_gen ) )
    plt.show()

    plt.figure()
