from typing import Any
import numpy as np 

from solver.moea.utils.utils_ea import dominance_min

class population: 

    def __init__(self): 
        pass

    def __call__(self, pop, F=None, G=None, ind=None) -> Any:
        if ind is None: 
            self.pop = pop 
            self.n_pop, self.n_var = self.pop.shape
            self.F = F
            self.G = G
        else: 
            self.pop[ind] = pop 
            if F is not None: self.F[ind] = F 
            if G is not None: self.G[ind] = G 

    def __getitem__(self, index):
        return self.pop[index]

    @property
    def X(self) -> np.ndarray: 
        return self.pop

    def evaluate(self, problem): 
        self.F = problem(self.pop)

    def update(self, pop): 
        self.pop = pop

    def parent_select(self, P): 
        return np.atleast_2d(self.pop[P])

class external_population: 

    def __init__(self):
        self.pop = None 
        self.F = None 

    def __call__(self, pop, pop_f):
        pop = np.atleast_2d(pop)
        pop_f = np.atleast_2d(pop_f)

        if self.pop is None: 
            self.pop = pop
            self.F = pop_f
        else: 
            for i in range(pop.shape[0]): self._update_par(pop[i], pop_f[i])

        self.n_pop = self.pop.shape[0]

    @property
    def non_dominate_sol(self): 
        return self.pop, self.F

    def _update_par(self, pop, pop_f): 
        add_ind = np.full((self.n_pop, ), False)
        for i in range(self.n_pop): 
            if dominance_min(self.F[i], pop_f): 
                add_ind[i] = True 
                break 

        if not add_ind.any(): 
            domi_ind = np.full((self.n_pop, ), False) 
            for i in range(self.n_pop): 
                if dominance_min(pop_f, self.F[i]): domi_ind[i] = True 

            self.pop = np.delete(self.pop, domi_ind, axis=0) 
            self.F = np.delete(self.F, domi_ind, axis=0) 

            self.pop = np.vstack([self.pop, pop])
            self.F = np.vstack([self.F, pop_f])
            


if __name__ == '__main__':
    
    from libmoon.problem.synthetic.zdt import ZDT1
    from utils_ea import population_initialization

    problem = ZDT1()
    pop = population(pop=population_initialization(10, problem))

    pop.evaluate(problem)

    print(pop.F)
    print()

    print()