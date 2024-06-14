import argparse
import numpy as np
from libmoon.util_global import synthetic_init, get_problem, uniform_pref
from libmoon.solver.gradient.methods import EPOSolver
from libmoon.solver.gradient.methods.pmgda_solver import PMGDASolver
from libmoon.solver.gradient.methods.base_solver import GradAggSolver
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt
from libmoon.util_global.constant import FONT_SIZE_2D, FONT_SIZE_3D, color_arr, beautiful_dict, root_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description= 'example script' )
    # Inv agg not very work
    parser.add_argument( '--agg', type=str, default='mtche')  # If solve is agg, then choose a specific agg method.
    parser.add_argument('--solver', type=str, default='agg')
    parser.add_argument( '--problem-name', type=str, default='VLMOP1')
    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--plt-pref-flag', type=str, default='N')
    parser.add_argument('--use-plt', type=str, default='Y')
    parser.add_argument('--PaperName', type=str, default='TETCI')
    # For PMGDA.
    parser.add_argument('--h-tol', type=float, default=1e-3)
    parser.add_argument('--sigma', type=float, default=0.9)
    parser.add_argument('--n-prob', type=int, default=8 )
    parser.add_argument('--n-iter', type=int, default=2000 )
    parser.add_argument('--seed-idx', type=int, default=0)
    args = parser.parse_args()

    if args.solver=='agg':
        args.task_name = '{}_{}'.format(args.solver, args.agg)
    else:
        args.task_name = args.solver

    print('Problem: {}'.format(args.problem_name))
    print('Task Name: {}'.format(args.task_name ))
    print('Seed: {}'.format(args.seed_idx))



    hv_seed = []
    seed_num = 3
    np.random.seed(args.seed_idx)
    problem = get_problem(problem_name=args.problem_name, n_var=10)
    if problem.n_obj == 2:
        args.n_prob = 8
    elif problem.n_obj == 3:
        args.n_prob = 15

    prefs = uniform_pref(n_prob=args.n_prob, n_obj = problem.n_obj, clip_eps=1e-2)

    if args.solver == 'epo':
        solver = EPOSolver(problem, step_size=1e-2, n_iter=args.n_iter, tol=args.tol )
    elif args.solver == 'pmgda':
        solver = PMGDASolver(problem, step_size=1e-2, n_iter=args.n_iter, tol=args.tol, sigma=args.sigma, h_tol=args.h_tol)
    elif args.solver == 'agg':
        solver = GradAggSolver(problem, step_size=1e-2, n_iter=args.n_iter, tol=args.tol, agg=args.agg)


    res = solver.solve( x=synthetic_init(problem, prefs), prefs=prefs )
    res['pref_mat'] = prefs
    # hv_seed.append( res['hv_history'] )
    sub_sample=1
    if problem.n_obj == 2:
        for idx in range(len(prefs)):
            plt.plot(res['y_history'][::sub_sample, idx, 0], res['y_history'][::sub_sample, idx, 1],
                    color=color_arr[idx])
    elif problem.n_obj == 3:
        ax = (plt.figure()).add_subplot(projection='3d')
        for idx in range( len(prefs) ) :
            ax.plot(res['y_history'][::sub_sample, idx, 0], res['y_history'][::sub_sample, idx, 1], res['y_history'][::sub_sample, idx, 2],
                    color=color_arr[idx])
        prefs_l2 = prefs / np.linalg.norm(prefs, axis=1, keepdims=True)
        for idx, pref in enumerate(prefs_l2):
            ax.scatter(pref[0], pref[1], pref[2], color=color_arr[idx], s=40)
        th1 = np.linspace(0, np.pi/2, 100)
        th2 = np.linspace(0, np.pi/2, 100)
        theta, phi = np.meshgrid(th1, th2)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)

        ax.plot_surface(x, y, z, alpha=0.3)
        ax.axis('equal')
        ax.view_init(30, 45)
        ax.set_xlim([0, 1.2])
        ax.set_ylim([0, 1.2])
        ax.set_zlim([0, 1.2])

        ax.set_xlabel('$L_1$', fontsize=FONT_SIZE_3D)
        ax.set_ylabel('$L_2$', fontsize=FONT_SIZE_3D)
        ax.set_zlabel('$L_3$', fontsize=FONT_SIZE_3D)


    fig_folder_name = os.path.join(root_name, args.PaperName, args.problem_name, '{}'.format(args.seed_idx))
    os.makedirs(fig_folder_name, exist_ok=True)
    fig_name = os.path.join(fig_folder_name, '{}.pdf'.format(args.task_name) )
    plt.savefig( fig_name )
    print('Save fig to {}'.format(fig_name) )
    plt.title( beautiful_dict[args.task_name] )

    import pickle
    pickle_folder = os.path.join(root_name, args.PaperName, args.task_name, 'M1', args.problem_name,
                               'seed_{}'.format(args.seed_idx), 'epoch_{}'.format(args.n_iter))
    os.makedirs(pickle_folder, exist_ok=True)
    pickle_name = os.path.join(pickle_folder, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        pickle.dump(res, f)
    print('Save pickle to {}'.format(pickle_name))