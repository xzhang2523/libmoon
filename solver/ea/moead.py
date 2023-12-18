import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

import argparse
import copy

from pymoo.util.ref_dirs import get_reference_directions

from solver.ea.utils import get_decomposition
from solver.ea.utils.population import population, external_population
from solver.ea.utils.termination import termination
from solver.ea.utils.genetic_operator import cross_sbx, mut_pm
from solver.ea.utils.utils_ea import population_initialization, neighborhood_selection


class MOEAD():
    def __init__(self,
                 mop: any = None,
                 ref_vec: np.ndarray = None,
                 n_neighbors: int = 10,
                 ):

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
              max_gen: int = 100,
              ) -> None:

        if mop is not None:
            self.mop = mop
        assert not mop.has_constraint
        self.termina.setup(max_gen=max_gen)

        if self.ref_vec is None:
            self.ref_vec = get_reference_directions("uniform", mop.get_number_objective, n_partitions=25)
        self.n_pop = len(self.ref_vec)

        pop = population_initialization(self.n_pop, self.mop)
        f = self.mop(pop)
        self.z_star = np.min(f, axis=0)

        self.pop(pop, f)
        self.termina(nfe=self.n_pop)

        self.neighbors = np.argsort(cdist(self.ref_vec, self.ref_vec), axis=1)[:, :self.n_neighbors]

        self.decomposition = get_decomposition('tch')

    def solve(self):

        while self.termina.has_next:
            self.step()

    def reset(self,
              problem: any):
        """
        Initialization
        """

    def step(self):

        for k in np.random.permutation(self.n_pop):
            P = neighborhood_selection(self.n_pop, self.neighbors[k])

            X = self.pop.parent_select(P)

            off_cross = cross_sbx(X, self.mop.get_lower_bound, self.mop.get_upper_bound)
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
                                                                                             self.z_star): self.pop(off,
                                                                                                                    F=off_f,




                                                                                                  ind=i)
if __name__ == '__main__':
    from problem.synthetic.zdt import ZDT1

    problem = ZDT1()
    alg = MOEAD()

    alg.setup(problem,max_gen=500)
    alg.solve()
    print('solving over')

    plt.scatter(alg.pop.F[:, 0], alg.pop.F[:, 1], c='none', edgecolors='r')
    plt.show()
