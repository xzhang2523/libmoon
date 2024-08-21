

import matplotlib.pyplot as plt
from torch.utils import data
from libmoon.problem.mtl.loaders import Adult, Credit, Compas
from libmoon.problem.mtl.objectives import from_name
from libmoon.problem.mtl.model_utils import model_from_dataset, dim_dict
from libmoon.problem.mtl.settings import adult_setting, credit_setting, compass_setting
import argparse
import torch
import numpy as np
from tqdm import tqdm
from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global.constant import color_arr, normalize_vec
from libmoon.util_global.grad_util import calc_gradients, flatten_grads
import os
from libmoon.util_global.constant import root_name

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--solver', type=str, default='mgda')
    parser.add_argument('--agg', type=str, default='tche')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-sub', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--update-counter', type=int, default=0)
    parser.add_argument('--uniform-update-counter', type=int, default=0)
    parser.add_argument('--uniform-update-iter', type=int, default=2000)
    # For pmgda
    parser.add_argument('--h-eps', type=float, default=5e-3)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    if args.solver == 'agg':
        args.task_name = 'agg_{}'.format(args.agg)
    else:
        args.task_name = args.solver

    output_folder_name = os.path.join(root_name, 'output', 'mtl', args.task_name, args.dataset,
                                      '{}'.format(args.seed))
    os.makedirs(output_folder_name, exist_ok=True)
    args.output_folder_name = output_folder_name
    if args.solver == 'uniform':
        from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreUniform
        core_uniform = CoreUniform()

    print( '{} on {}'.format(args.task_name, args.dataset) )
    dataset_dict = {'adult':Adult(split="train"),
                    'credit':Credit(split="train"),
                    'compass':Compas(split="train")}

    dataset = dataset_dict[args.dataset]

    setting_dict = {'adult':adult_setting,
                    'credit':credit_setting,
                    'compass':compass_setting}

    settings = setting_dict[args.dataset]
    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    obj_arr = from_name(settings['objectives'], dataset.task_names())
    model_arr = [model_from_dataset(args.dataset, dim=dim_dict[args.dataset])
                 for _ in range(args.n_sub)]


    optimizer_arr = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_arr]
    pref_mat = torch.Tensor(uniform_pref(n_prob=args.n_sub, clip_eps=1e-2))
    epoch_loss_pref = []
    # For seperate models, we have N independent models, each with its own optimizer.
    for _ in tqdm( range(args.epoch) ):
        loss_batch = []
        for b, batch in enumerate(trainloader):
            loss_mat = [0] * len(pref_mat)
            for pref_idx, pref in enumerate(pref_mat):
                # data.keys() : dict_keys(['data', 'labels', 'sensible_attribute'])
                logits = model_arr[pref_idx](batch)
                batch.update(logits)
                loss_vec = [0] * 2
                for idx, obj in enumerate(obj_arr) :
                    loss_vec[idx] = obj( **batch )
                loss_vec = torch.stack(loss_vec)
                loss_vec = normalize_vec(loss_vec, problem=args.dataset)
                # Calc the gradient, and get the alpha.
                if args.solver != 'agg':
                    gradients, obj_values = calc_gradients(batch, model_arr[pref_idx], obj_arr)
                    Jacobian = torch.stack( [flatten_grads(gradients[idx]) for idx in range(2)] )

                    if args.solver == 'epo':
                        from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreEPO
                        core_epo = CoreEPO(pref)
                        alpha = torch.Tensor( core_epo.get_alpha(Jacobian, loss_vec) )

                    elif args.solver == 'mgda':
                        from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreMGDA
                        core_mgda = CoreMGDA()
                        alpha = torch.Tensor( core_mgda.get_alpha(Jacobian) )

                    elif args.solver == 'pmgda':
                        from libmoon.solver.gradient.methods.bk.core_solver_bk import CorePMGDA
                        from libmoon.solver.gradient.methods.pmgda_core import get_nn_pmgda_componets
                        h_val, Jhf = get_nn_pmgda_componets(loss_vec, pref, args)
                        grad_h = torch.Tensor(Jhf) @ Jacobian
                        core_pmgda = CorePMGDA(args)
                        alpha = core_pmgda.get_alpha(Jacobian, grad_h, h_val, args, return_coeff=True, Jhf=Jhf)
                        alpha = torch.Tensor( alpha )

                if args.solver == 'agg':
                    agg_func = agg_dict[args.agg]
                    scalar_loss = torch.squeeze(agg_func(loss_vec.unsqueeze(0), pref.unsqueeze(0)))

                elif args.solver == 'uniform':
                    agg_func = agg_dict['mtche']
                    scalar_loss = torch.squeeze(agg_func(loss_vec.unsqueeze(0), pref.unsqueeze(0)))

                else:
                    scalar_loss = torch.sum(alpha * loss_vec)

                optimizer_arr[pref_idx].zero_grad()
                scalar_loss.backward()
                optimizer_arr[pref_idx].step()     # Use a counter to
                args.update_counter += 1
                loss_mat[pref_idx] = loss_vec.detach().cpu().numpy()

            loss_mat = np.array( loss_mat )
            loss_batch.append( loss_mat )

            if args.solver == 'uniform':
                if args.update_counter % args.uniform_update_iter == 0:
                    pref_mat = core_uniform.update_pref_mat(pref_mat, loss_mat, args)
                    pref_mat = torch.clamp(pref_mat, 1e-2, 1-1e-2)

        loss_batch = np.array(loss_batch)
        epoch_loss_pref.append( np.mean(loss_batch, axis=0) )

    epoch_loss_pref = np.array(epoch_loss_pref)
    epoch_loss_pref_final = epoch_loss_pref[-1,:,:]



    fig = plt.figure()
    for idx in range(args.n_sub):
        plt.plot(epoch_loss_pref[:,idx, 0], linewidth=2, color=color_arr[idx])
        plt.plot(epoch_loss_pref[:,idx, 1], linewidth=2, color=color_arr[idx], linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    fig_name = os.path.join(output_folder_name, 'learning_curve.pdf')
    print('Saving figure to', fig_name)
    plt.savefig( fig_name )

    fig = plt.figure()
    plt.scatter(epoch_loss_pref_final[:,0], epoch_loss_pref_final[:,1], color='black', s=50)
    rho_max = np.max( np.linalg.norm(epoch_loss_pref_final, axis=1) )

    pref_mat_np = pref_mat.detach().cpu().numpy()
    pref_mat_np = pref_mat_np / np.linalg.norm(pref_mat_np, axis=1, keepdims=True) * rho_max

    for pref in pref_mat_np:
        plt.plot([0, pref[0]], [0, pref[1]], color='grey', linestyle='--')

    plt.xlabel('$L_1$')
    plt.ylabel('$L_2$')

    fig_name = os.path.join(output_folder_name, 'res.pdf')
    print('Saving figure to', fig_name)
    plt.savefig(fig_name)

    # Finally save indicators
    from libmoon.metrics.metrics import compute_indicators
    indicators_dict = compute_indicators(epoch_loss_pref_final)
    indicator_name = os.path.join(output_folder_name, 'indicators.txt')

    with open(indicator_name, 'w') as f:
        for k,v in indicators_dict.items():
            f.write('{}: {}\n'.format(k, v))
        f.write('{}: {}\n'.format('counter', args.update_counter ))
    print('Saving indicators to', indicator_name)

    # Finally, save all information into pickle files.
    pickle_name = os.path.join(output_folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        import pickle
        pickle.dump( {
            'args': args,
            'epoch_loss_pref': epoch_loss_pref,
            'epoch_loss_pref_final': epoch_loss_pref_final,
            'pref_mat_np': pref_mat_np,
            'indicators_dict': indicators_dict
        }, f)

    print('Saving all information to', pickle_name)