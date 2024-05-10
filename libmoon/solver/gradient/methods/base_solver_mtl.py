import torch
from libmoon.problem.mtl.mnist import MultiMnistProblem
import argparse


from libmoon.util_global.weight_factor import uniform_pref

class BaseSolverMTL:
    def __init__(self):
        print()

class BaseSolverMTLAgg:
    def __init__(self):
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example script')
    parser.add_argument('--n-partition', type=int, default=10)
    parser.add_argument('--agg', type=str, default='ls')  # If solve is agg, then choose a specific agg method.
    parser.add_argument('--solver', type=str, default='agg')
    # ['agg', 'epo', 'moosvgd', 'hvgrad', 'pmtl', 'mgda']
    parser.add_argument('--problem-name', type=str, default='VLMOP2')
    parser.add_argument('--iter', type=int, default=2000)

    parser.add_argument('--step-size', type=float, default=0.1)
    parser.add_argument('--batch-size', type=float, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--plt-pref-flag', type=str, default='N')
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    prefs = uniform_pref(n_partition=4)
    args.n_prob = len(prefs)

    problem = MultiMnistProblem( args )
    solver = BaseSolverMTLAgg()
    print()

