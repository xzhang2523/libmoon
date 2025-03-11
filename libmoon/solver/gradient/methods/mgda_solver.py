import torch.autograd
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
from libmoon.problem.synthetic import ZDT1, ZDT2
from libmoon.solver.gradient.methods.core.core_solver import MGDAUBCore
from matplotlib import pyplot as plt
'''
    MGDA solver, published in: 
    1. Multiple-gradient descent algorithm (MGDA) for multiobjective optimizationAlgorithme de descente à gradients multiples pour lʼoptimisation multiobjectif
    2. Sener, Ozan, and Vladlen Koltun. "Multi-task learning as multimnist-objective optimization." Advances in neural information processing systems 31 (2018).
'''
class MGDAUBSolver(GradBaseSolver):
    def __init__(self, problem, prefs, step_size=1e-3, n_epoch=500, tol=1e-3, folder_name=None):
        self.folder_name = folder_name
        self.mgda_core = MGDAUBCore()
        self.problem = problem
        self.prefs = prefs
        self.solver_name = 'MGDA'

        super().__init__(step_size, n_epoch, tol, self.mgda_core)

    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)



if __name__ == '__main__':
    n_prob = 10
    n_var=10
    problem = ZDT1(n_var=n_var)
    solver = MGDAUBSolver(0.1, 1000, 1e-6)
    x = torch.rand(n_prob, n_var)
    prefs = torch.rand(n_prob, 2)
    res = solver.solve(problem, x, prefs)
    y_arr = res['y']
    plt.scatter(y_arr[:, 0], y_arr[:, 1])
    plt.show()

