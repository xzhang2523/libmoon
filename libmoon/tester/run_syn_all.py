from libmoon.solver.gradient.methods import MGDAUBSolver, EPOSolver
from libmoon.problem.synthetic.zdt import ZDT1

import argparse




if __name__ == '__main__':
    n_prob = 10
    n_var=10

    problem = ZDT1(n_var=n_var)
    solver = MGDAUBSolver(step_size=1e-2, max_iter=10, tol=1e-2)
