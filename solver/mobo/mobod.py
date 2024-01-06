import numpy as np
from smt.surrogate_models import KRG

from scipy.spatial.distance import cdist

from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from pyDOE3 import lhs

from solver.mobo.utils.termination import termination


'''
    Main algorithm framework for  Decomposition-based Multi-objective Bayesian Optimization.
'''



class MOBOD(object):
    def __init__(self,
                 mop: any = None,
                 ref_vecs: np.ndarray = None,
                 max_iter: int = None,
                 batch_size: int = 5,
                 ):
        self.mop = mop
        self.ref_vecs = ref_vecs
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.termina = termination()

    def setup(self,
              mop: any = None,
              ref_vec: np.ndarray = None,
              max_iter: int = None,
              batch_size: int = 5,
              ) -> None:
        if mop is not None:
            self.mop = mop
            self.n_var, self.n_obj = mop.n_var, mop.n_obj
        assert not mop.has_constraint
        self.termina.setup(max_gen=max_iter)

        if self.ref_vecs is None:
            self.ref_vecs = get_reference_directions("uniform", mop.get_number_objective, n_partitions=199)
        self.pop_size = len(self.ref_vecs)
        # to keep track of data
        self.X = None
        self.Y = None
        self.sample_num = 0
        self.i_iter = 0
        self.pf_idx = None  # idx of nondominated solutions
        self.ref_point = None
        self.hv_all_value = np.zeros([max_iter + 1, 1])

    def construct_model(self, X_obs, Y_obs):
        # build surrogate models
        theta = [self.sample_num ** (-1. / self.sample_num)] * self.n_var
        self.surrogate_model = [KRG(theta0=theta) for i in range(self.n_obj)]
        for i in range(self.n_obj):
            self.surrogate_model[i].options.__setitem__('print_global', False)
            self.surrogate_model[i].set_training_values(X_obs, Y_obs[:, i])
            self.surrogate_model[i].train()

    def predict_model(self, X):
        N = X.shape[0]
        u = np.zeros(shape=(N, self.n_obj), dtype=float)
        MSE = np.zeros(shape=(N, self.n_obj), dtype=float)

        for j in range(self.n_obj):
            u[:, j] = self.surrogate_model[j].predict_values(X)[:, 0]
            # MSE[:,j] = np.sqrt(model[j].predict_variances(PopDec))
            MSE[:, j] = self.surrogate_model[j].predict_variances(X)[:, 0]

        MSE[MSE < 0] = 0
        s = np.sqrt(MSE)
        return u, s

    def evaluator_acquisition(self, u, sigma, ref_vec, pref_inc):
        pass

    def aggregate_data(self, X_new, Y_new):
        if self.sample_num == 0:
            self.X = X_new.copy()
            self.Y = Y_new.copy()
        else:
            self.X = np.vstack([self.X, X_new])
            self.Y = np.vstack([self.Y, Y_new])
        self.sample_num += len(X_new)

        # nondominated X, Y
        nds = NonDominatedSorting()
        self.pf_idx = nds.do(self.Y)
        # X_nds = self.X[self.pf_idx[0]]
        Y_nds = self.Y[self.pf_idx[0]]

        hv = HV(ref_point=self.ref_point)
        hv_value = hv(Y_nds)
        self.hv_all_value[self.i_iter, 0] = hv_value

    def step(self):
        pass

    def solve(self, X_init, Y_init):
        if self.ref_point is None:
            self.ref_point = np.max(Y_init, axis=0)
        self.aggregate_data(X_init, Y_init)

        s = str('n_iter').center(12) + " | " + str('n_eval').center(12) + " | " + str('HV').center(12)
        print("=" * len(s))
        print(s)
        print("=" * len(s))
        print(str(self.i_iter).center(12), "|", str(self.sample_num).center(12), "| ",
              f"%.{10}f" % self.hv_all_value[self.i_iter, 0])

        while self.termina.has_next:
            self.i_iter += 1
            # generate new samples
            X_next = self.step()  # check
            Y_next = self.mop.evaluate(X_next)
            self.termina(nfe=self.batch_size, gen=1)
            self.aggregate_data(X_next, Y_next)
            print(str(self.i_iter).center(12), "|", str(self.sample_num).center(12), "| ",
                  f"%.{10}f" % self.hv_all_value[self.i_iter, 0])

        print("=" * len(s))

    def MOEAD_GR_(self, ref_vecs, pref_incs):
        # using MOEA/D-GR to solve subproblems
        maxIter = 50
        N = self.pop_size
        T = int(np.ceil(N / 10))  # size of neighbourhood: 0.1*N
        B = np.argsort(cdist(ref_vecs, ref_vecs), axis=1, kind='quicksort')[:, :T]

        # the initial population for MOEA/D

        Pop_Dec = (self.mop.get_upper_bound - self.mop.get_lower_bound) * lhs(self.n_var,
                                                                              samples=N) + self.mop.get_lower_bound
        Pop_u, Pop_sigma = self.predict_model(Pop_Dec)
        Pop_EID = self.evaluator_acquisition(Pop_u, Pop_sigma, ref_vecs, pref_incs)

        # optimization
        for gen in range(maxIter - 1):
            for i in range(N):
                if np.random.random() < 0.8:  # delta
                    P = B[i, np.random.permutation(B.shape[1])]
                else:
                    P = np.random.permutation(N)
                # generate an offspring 1*d
                Off_Dec = self._OperatorDE(Pop_Dec[i, :][None, :], Pop_Dec[P[0], :][None, :], Pop_Dec[P[1], :][None, :])
                Off_u, Off_sigma = self.predict_model(Off_Dec)
                # Global Replacement  MOEA/D-GR
                # Find the most approprite subproblem and its neighbourhood
                EID_all = self.evaluator_acquisition(np.repeat(Off_u, N, axis=0), np.repeat(Off_sigma, N, axis=0),
                                                     ref_vecs, pref_incs)
                best_index = np.argmax(EID_all)
                P = B[best_index, :]  # replacement neighborhood

                offindex = P[Pop_EID[P] < EID_all[P]]
                if len(offindex) > 0:
                    Pop_Dec[offindex, :] = np.repeat(Off_Dec, len(offindex), axis=0)
                    Pop_u[offindex, :] = np.repeat(Off_u, len(offindex), axis=0)
                    Pop_sigma[offindex, :] = np.repeat(Off_sigma, len(offindex), axis=0)
                    Pop_EID[offindex] = EID_all[offindex]

        return Pop_Dec, Pop_u, Pop_sigma

    def _OperatorDE(self, Parent1, Parent2, Parent3):
        '''
            generate one offspring by P1 + 0.5*(P2-P3) and polynomial mutation.
        '''
        # Parameter
        CR = 1
        F = 0.5
        proM = 1
        disM = 20
        #
        N, D = Parent1.shape
        # Differental evolution
        Site = np.random.rand(N, D) < CR
        Offspring = Parent1.copy()
        Offspring[Site] = Offspring[Site] + F * (Parent2[Site] - Parent3[Site])
        # Polynomial mutation
        Lower = np.atleast_2d(self.mop.get_lower_bound)  # numpy  Upper=np.array(Upper)[None,:]
        Upper = np.atleast_2d(self.mop.get_upper_bound)  # Lower = np.atleast_2d(Lower)
        U_L = Upper - Lower
        Site = np.random.rand(N, D) < proM / D
        mu = np.random.rand(N, D)
        temp = np.logical_and(Site, mu <= 0.5)
        Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        delta1 = (Offspring - Lower) / U_L
        delta2 = (Upper - Offspring) / U_L
        #  mu <= 0.5
        val = 2. * mu + (1 - 2. * mu) * (np.power(1. - delta1, disM + 1))
        Offspring[temp] = Offspring[temp] + (np.power(val[temp], 1.0 / (disM + 1)) - 1.) * U_L[temp]
        # mu > 0.5
        temp = np.logical_and(Site, mu > 0.5)
        val = 2. * (1.0 - mu) + 2. * (mu - 0.5) * (np.power(1. - delta2, disM + 1))
        Offspring[temp] = Offspring[temp] + (1.0 - np.power(val[temp], 1.0 / (disM + 1))) * U_L[temp]

        return Offspring