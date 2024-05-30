from libmoon.problem.mtl.core.pref_mtl import MTL_EPO_Solver
from matplotlib import pyplot as plt




if __name__ == '__main__':
    dataset_name = 'adult'
    n_prob = 5

    epo_solver = MTL_EPO_Solver(dataset_name, n_prob)
    res = epo_solver.solve()
    plt.scatter(res[:,0], res[:,1])

    plt.show()