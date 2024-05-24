import numpy as np
from libmoon.solver.gradient.methods import MGDASolver, GradAggSolver, EPOSolver, MOOSVGDSolver, GradHVSolver, PMTLSolver
from libmoon.solver.gradient.methods import PMGDASolver, UniformSolver

from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global.constant import get_problem
from libmoon.util_global.initialization import synthetic_init

from libmoon.visulization.view_res import vedio_res
import argparse
import torch
from matplotlib import pyplot as plt
import pickle
import time
from libmoon.util_global.constant import root_name
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script' )
    parser.add_argument( '--n-partition', type=int, default=10 )
    # Inv agg not very work
    parser.add_argument( '--agg', type=str, default='tche')  # If solve is agg, then choose a specific agg method.
    parser.add_argument('--solver', type=str, default='epo')
    parser.add_argument( '--problem-name', type=str, default='VLMOP2')

    parser.add_argument('--iter', type=int, default=2000)
    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--plt-pref-flag', type=str, default='N')
    parser.add_argument('--use-plt', type=str, default='Y')
    # For Pmgda
    parser.add_argument('--h-eps', type=float, default=1e-2)
    parser.add_argument('--sigma', type=float, default=0.8)


    args = parser.parse_args()
    problem = get_problem(args.problem_name)

    args.n_obj, args.n_var = problem.n_obj, problem.n_var

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
        solver = PMGDASolver(args.step_size, args.iter, args.tol)
    elif args.solver=='uniform':
        solver = UniformSolver(args.step_size, args.iter, args.tol)
    else:
        raise Exception('solver not supported')

    if args.solver == 'agg':
        args.task_name = '{}_{}'.format(args.solver, args.agg)
    else:
        args.task_name = args.solver

    if args.solver == 'agg':
        args.folder_name = os.path.join(root_name, 'output', 'discrete', args.task_name, args.problem_name)
    else:
        args.folder_name = os.path.join(root_name, 'output', 'discrete', args.task_name, args.problem_name)


    os.makedirs(args.folder_name, exist_ok=True)
    prefs = uniform_pref( args.n_partition, problem.n_obj, clip_eps=1e-2)

    args.n_prob = len(prefs)
    ts = time.time()

    problem_name = 'VLMOP2'
    # problem = get_problem(problem_name=problem_name)
    res = solver.solve( get_problem(problem_name=problem_name), x=synthetic_init(problem_name, prefs), prefs=prefs, args=args)
    print()

    # plt.plot(res['hv_arr'])
    # plt.xlabel('Iteration')
    # plt.ylabel('HV')
    # plt.title('HV curve')
    #
    # fig_name = os.path.join(args.folder_name, 'hv_curve.pdf')
    # plt.savefig(fig_name)
    # print('Save hv curve fig to %s' % fig_name)
    #
    # elapsed = time.time() - ts
    # res['elapsed'] = elapsed
    #
    # use_fig = True
    # if use_fig:
    #     fig = plt.figure()
    #     plt.scatter(res['y'][:,0], res['y'][:,1], label='Solutions')
    #     plt.plot(problem.get_pf()[:,0], problem.get_pf()[:,1], label='PF')
    #
    #     prefs_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
    #     for elem in prefs_l2:
    #         plt.plot([0, elem[0]], [0, elem[1]], color='gray', linestyle='dashed')
    #
    #     plt.legend(fontsize=16)
    #     plt.xlabel('$L_1$', fontsize=18)
    #     plt.ylabel('$L_2$', fontsize=18)
    #
    #     fig_name = os.path.join(args.folder_name, 'res.pdf')
    #     plt.savefig(fig_name)
    #     print('Save fig to %s' % fig_name)
    #     if args.use_plt == 'Y':
    #         plt.show()
    #
    #
    # use_vedio = False
    # if use_vedio:
    #     vedio_res(res, problem, prefs, args)
    #
    # pickle_name = os.path.join(args.folder_name, 'res.pkl')
    # with open(pickle_name, 'wb') as f:
    #     pickle.dump(res, f)
    #
    # print('Save pickle to %s' % pickle_name)
