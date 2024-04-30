from .base_solver import GradBaseSolver



class PMGDASolver(GradBaseSolver):
    # Related paper:

    # http://arxiv.org/abs/2402.09492.

    def __init__(self, step_size, max_iter, tol):
        print('pmgda solver')
        print()

        super().__init__(step_size, max_iter, tol)

    def solve(self, problem, x, prefs, args):
        print()

