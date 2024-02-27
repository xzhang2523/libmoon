# -*- coding: utf-8 -*-
"""
------------------------------- Reference --------------------------------
 L. Zhao and Q. Zhang, Hypervolume-Guided Decomposition for Parallel
 Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary
 Computation, 2023.
"""
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from solver.mobo.mobod import MOBOD


class DirHVEGO(MOBOD):
    def __init__(self,
                 mop: any = None,
                 ref_vecs: np.ndarray = None,
                 max_iter: int = None,
                 batch_size: int = 5
                 ) -> None:
        super().__init__(mop, ref_vecs, max_iter, batch_size)

    def evaluator_acquisition(self, u, sigma, ref_vec, pref_inc):
        '''
        Parameters:
        ----------
        u :
        sigma :
        ref_vec: direction vector
        pref_inc :  preference-conditional incumbent

        Returns
        -------
        EI_D : preference-conditional EI: DirHV-EI(X|pref_vec)

        '''
        xi_minus_u = pref_inc - u  # N*M
        tau = xi_minus_u / sigma  # N*M
        temp = xi_minus_u * stats.norm(0, 1).cdf(tau) + sigma * stats.norm(0, 1).pdf(tau)  # N*M
        EI_D = np.prod(temp, axis=1)
        return EI_D

    def step(self):
        #  normalization
        X_norm = self.X.copy()
        Y_scaler = MinMaxScaler(feature_range=(0, 1))
        Y_norm = Y_scaler.fit_transform(self.Y)

        # GP modeling
        self.construct_model(X_norm, Y_norm)

        # Utopian point
        Z = -0.01 * np.ones(shape=[1, self.n_obj])
        # Calculate the Intersection points and Direction vectors
        Xi, Lambda = self.get_Xi(Y_norm[self.pf_idx[0]], self.ref_vecs, Z)
        # Use MOEA/D to maximize DirHV-EI
        Pop_Dec, Pop_u, Pop_s = self.MOEAD_GR_(Lambda, Xi)
        # Discard duplicate candidates
        PopDec, ia = np.unique(Pop_Dec, axis=0, return_index=True)
        Pop_u = Pop_u[ia, :]
        Pop_s = Pop_s[ia, :]
        N = self.ref_vecs.shape[0]  # pop size

        # Compute EI_D for all the points in Q
        L = PopDec.shape[0]
        EIDs = np.zeros((L, N))
        for j in range(L):
            EIDs[j, :] = self.evaluator_acquisition(np.tile(Pop_u[j], (N, 1)), np.tile(Pop_s[j], (N, 1)), Lambda, Xi)

        # Find q solutions with the greedy algorithm
        # Batch_size = np.min(Problem.maxFE - Problem.FE, q)  # the total budget is Problem.maxFE
        Qb = self.batchSelection(EIDs, self.batch_size)

        return PopDec[Qb, :]

    def batchSelection(self, EIDs, q):
        # Algorithm 3: Submodularity-based Batch Selection
        L, N = EIDs.shape
        Qb = []
        temp = EIDs.copy()
        beta = np.zeros(N)
        for i in range(q):
            index = np.argmax(np.sum(temp, axis=1))
            Qb.append(index)
            beta = beta + temp[index, :]
            # temp: [EI_D(x|\lambda) - beta]_+
            temp = EIDs - np.repeat(beta[None, :], L, axis=0)
            temp[temp < 0] = 0
        return Qb

    def get_Xi(self, A, W, Z):
        N = self.ref_vecs.shape[0]
        W_ = 1.1 * self.ref_vecs - Z
        Lambda = W_ / np.linalg.norm(W_, axis=1, keepdims=True)
        # Eq. 11, compute the intersection points
        Lambda_ = 1.0 / Lambda
        A = A - Z  # L*M
        G = np.outer(Lambda_[:, 0], A[:, 0])  # N*L, f1
        for j in range(1, self.n_obj):
            G = np.maximum(G, np.outer(Lambda_[:, j], A[:, j]))  # N*L, max(fi,fj)

        # minimum of mTch for each direction vector
        Lmin = np.min(G, axis=1, keepdims=True)  # N*1  one for each direction vector

        # N*M  Intersection points
        Xi = Z + np.multiply(Lmin, Lambda)

        return Xi, Lambda





if __name__ == '__main__':
    from libmoon.problem.synthetic.zdt import ZDT1
    from pyDOE3 import lhs

    prob = ZDT1(n_var=8,
                n_obj=2,
                lower_bound=np.array([.0] * 8),
                upper_bound=np.array([1.] * 8))
    alg = DirHVEGO()
    alg.setup(prob, max_iter=10, batch_size=5)

    xdoe = (prob.ub - prob.lb) * lhs(prob.n_var, samples=11 * prob.n_var - 1,
                                     criterion='maximin', iterations=10) + prob.lb


    ydoe = prob.evaluate(xdoe)

    alg.solve(xdoe, ydoe)

    plt.scatter(alg.Y[alg.pf_idx[0], 0], alg.Y[alg.pf_idx[0], 1], c='none', edgecolors='r')
    plt.show()