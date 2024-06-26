import numpy as np
from libmoon.problem.mtl.core.pref_set_mtl import MTL_Pref_Solver
from libmoon.util_mtl.util import get_mtl_prefs
from matplotlib import pyplot as plt
import os
import argparse
import torch
from libmoon.metrics.metrics import compute_inner_product, compute_cross_angle
from libmoon.metrics.metrics import compute_indicators, compute_hv
from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global import color_arr
from libmoon.util_global.constant import beautiful_dict
import pandas as pd
import pickle
from libmoon.util_global.constant import root_name, get_param_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='M1')
    parser.add_argument('--dataset-name', type=str, default='compass')
    parser.add_argument('--PaperName', type=str, default='TETCI')
    parser.add_argument('--solver', type=str, default='pmgda')       # Valid solvers: ['PMTL', 'MOOSVGD', 'HVGrad'].
    parser.add_argument('--agg', type=str, default='cosmos')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-prob', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--cosmos-hp', type=float, default=20.0)

    # Adult dataset has 335 iterations. So 400 epoch has 4 uniform update.
    # Compass dataset has 210 iterations.
    parser.add_argument('--uniform-update-iter', type=int, default=67500)

    uniform_update_iter_dict = {
        'adult': 67500,
        'credit': 67500,
        'compass': 2100,
    }

    parser.add_argument('--uniform-pref-update', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=512)

    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    print('cosmos-hp: {}'.format(args.cosmos_hp))
    # Here, we should calculate how many update per epoch we have.
    if args.dataset_name == 'credit':
        args.batch_size=128

    args.uniform_update_iter = uniform_update_iter_dict[args.dataset_name]

    if args.dataset_name in ['mnist', 'fashion', 'fmnist']:
        sigma = 0.6
    else:
        sigma = args.sigma


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {
        'batch_size' : args.batch_size,
        'lr' : args.lr,
        'epoch' : args.epoch,
        'solver' : 'pmgda',
        'dataset_name' : args.dataset_name,
        'obj_normalization' : False,
        'n_prob' : args.n_prob,
        'sigma' : sigma,
        'h_tol' : 1e-3,
        'device' : device,
        'uniform_update_iter' : args.uniform_update_iter,
        'uniform_pref_update': args.uniform_pref_update,
    }

    kwargs.update( vars(args) )
    kwargs.pop('PaperName')

    if kwargs['solver'] == 'agg':
        task_name = '{}_{}'.format(kwargs['solver'], kwargs['agg'])
    else:
        task_name = kwargs['solver']

    np.random.seed(kwargs['seed'])
    print('Task name: {} on seed {}'.format(task_name, args.seed))
    print('Dataset:{}'.format(kwargs['dataset_name']))

    if torch.cuda.is_available():
        print('Using GPU')
    else:
        print('Using CPU')
    folder_name = os.path.join( root_name, args.PaperName, task_name,
                               kwargs['architecture'], '{}'.format(args.dataset_name), 'seed_{}'.format(args.seed), 'epoch_{}'.format(args.epoch) )

    os.makedirs(folder_name, exist_ok=True)
    kwargs['folder_name'] = folder_name


    mtl_solver = MTL_Pref_Solver(**kwargs)
    pref_mat = get_mtl_prefs(args.dataset_name, kwargs['n_prob'],
                             obj_normalization=kwargs['obj_normalization'])

    res, res_history, pref_mat = mtl_solver.solve(pref_mat)

    pref_mat = pref_mat.cpu().numpy()
    os.makedirs(folder_name, exist_ok=True)
    fig = plt.figure()

    for idx, res_elem in enumerate(res):
        plt.scatter(res_elem[0], res_elem[1], color=color_arr[idx], label='Pref. {}'.format(idx+1))

    pref_mat_l2 = pref_mat / np.linalg.norm(pref_mat, axis=1, keepdims=True)

    rho = np.max(np.linalg.norm(res, axis=1))
    for idx, pref in enumerate(pref_mat_l2):
        plt.plot([0, pref[0]*rho], [0, pref[1]*rho], color=color_arr[idx] )

    plt.xlabel('$L_1$', fontsize=18)
    plt.ylabel('$L_2$', fontsize=18)
    plt.legend(fontsize=12)
    fig_name = os.path.join(folder_name, '{}.pdf'.format('fig') )
    plt.savefig( fig_name )

    indicator_dict = compute_indicators(res, pref_mat)
    print('Save fig to {}'.format(fig_name))

    fig = plt.figure()
    hv_history = []
    for res_elem in res_history:
        hv_history.append(compute_hv(res_elem))

    cross_angle_history = []
    for res_elem in res_history:
        cross_angle_history.append(compute_cross_angle(res_elem, pref_mat))

    plt.plot(hv_history)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('HV', fontsize=18)

    fig_name = os.path.join(folder_name, '{}.pdf'.format('hv') )
    plt.savefig( fig_name )
    print('Save fig to {}'.format(fig_name))

    indicator_txt_name = os.path.join(folder_name, 'indicator.csv')
    df = pd.DataFrame(indicator_dict, index=[0])
    df.to_csv(indicator_txt_name, index=False)
    print('Save indicator to {}'.format(indicator_txt_name))

    pickle_file_name = os.path.join(folder_name, 'res.pickle')

    with open(pickle_file_name, 'wb') as f:
        pickle.dump({
            'res' : res,
            'res_history' : res_history,
            'pref_mat' : pref_mat,
            'hv_history' : hv_history,
            'cross_angle_history': cross_angle_history,
        }, f)
    print('Save pickle to {}'.format(pickle_file_name))

    # save all the pickle to a text file
    text_name = os.path.join(folder_name, 'hyperparam.txt')
    with open(text_name, 'w') as f:
        for key, val in kwargs.items():
            f.write('{}: {}\n'.format(key, val))
    print('Save hyperparam to {}'.format(text_name))
