import numpy as np
import os
import sys
print(f"vscode current run path is {os.getcwd()}")
os.chdir(sys.path[0])
print(f"set  py path as current path ")
print(f"vscode current run path is {os.getcwd()}")

from libmoon.solver.gradient import MGDASolver, GradAggSolver, EPOSolver, MOOSVGDSolver, GradHVSolver, PMTLSolver
from libmoon.util_global.weight_factor.funs import uniform_pref
from libmoon.util_global.constant import problem_dict
from libmoon.visulization.view_res import vis_res, vedio_res
import argparse
import torch
from matplotlib import pyplot as plt
import pickle
import time




if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script' )
    parser.add_argument( '--n-partition', type=int, default=10 )
    parser.add_argument( '--agg', type=str, default='tche')  # If solve is agg, then choose a specific agg method.
    parser.add_argument('--solver', type=str, default='hvgrad')
    # ['agg', 'epo', 'moosvgd', 'hvgrad', 'pmtl', 'mgda']
    parser.add_argument( '--problem-name', type=str, default='VLMOP2')
    parser.add_argument('--iter', type=int, default=2000)
    parser.add_argument('--step-size', type=float, default=0.1)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--plt-pref-flag', type=str, default='N')

    args = parser.parse_args()
    problem = problem_dict[args.problem_name]
    args.n_obj, args.n_var = problem.n_obj, problem.n_var
    root_name = os.path.dirname(os.path.dirname(__file__))


    if args.solver == 'mgda':
        solver = MGDASolver(args.step_size, args.iter, args.tol)
    elif args.solver == 'agg':
        solver = GradAggSolver(args.step_size, args.iter, args.tol)
    elif args.solver == 'epo':
        solver = EPOSolver(args.step_size, args.iter, args.tol)
    elif args.solver == 'moosvgd':
        solver = MOOSVGDSolver(args.step_size, args.iter, args.tol)
    elif args.solver == 'hvgrad':
        solver = GradHVSolver(args.step_size, args.iter, args.tol)
    elif args.solver == 'pmtl':
        solver = PMTLSolver(args.step_size, args.iter, args.tol)
    elif args.solver=='pmgda':
        assert False, 'will be implemented soon'
    else:
        raise Exception('solver not supported')

    if args.solver == 'agg':
        args.folder_name = os.path.join(root_name, 'output', args.problem_name, '{}_{}'.format(args.solver, args.agg))
    else:
        args.folder_name = os.path.join(root_name, 'output', args.problem_name, args.solver)

    os.makedirs(args.folder_name, exist_ok=True)
    prefs = uniform_pref( args.n_partition, problem.n_obj, clip_eps=1e-2)

    args.n_prob = len(prefs)
    if 'lbound' in dir(problem):
        if args.problem_name == 'VLMOP1':
            x0 = torch.rand(args.n_prob, problem.n_var) * 2 / np.sqrt(problem.n_var) - 1 / np.sqrt(problem.n_var)
        else:
            x0 = torch.rand(args.n_prob, problem.n_var)
    else:
        x0 = torch.rand( args.n_prob, problem.n_var ) * 20 - 10


    ts = time.time()
    res = solver.solve( problem, x=x0, prefs=prefs, args=args)

    elapsed = time.time() - ts
    res['elapsed'] = elapsed


    use_fig=False
    if use_fig:
        vis_res(res, problem, prefs, args)
        fig_name = os.path.join(args.folder_name, 'res.svg')
        plt.savefig(fig_name)
        print('Save fig to %s' % fig_name)
        plt.show()


    use_vedio=True
    if use_vedio:
        vedio_res(res, problem, prefs, args)


    pickle_name = os.path.join(args.folder_name, 'res.pkl')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)

    print('Save pickle to %s' % pickle_name)
