import matplotlib.pyplot as plt
from torch.utils import data
from libmoon.problem.mtl.loaders import Adult, Credit, Compas
from libmoon.problem.mtl.objectives import from_name
from libmoon.problem.mtl.model_utils import model_from_dataset, dim_dict
from libmoon.problem.mtl.settings import adult_setting, credit_setting, compass_setting
# To achieve discrete solutions.
import argparse
import torch
import numpy as np
from tqdm import tqdm
from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global.constant import color_arr, normalize_vec
from libmoon.util_global.grad_util import calc_gradients, flatten_grads




if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--solver', type=str, default='hvgrad')
    parser.add_argument('--agg', type=str, default='ls')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-sub', type=int, default=5)    # Denoted as K in the paper.
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-obj', type=int, default=2)
    # For pmgda.
    parser.add_argument('--h-eps', type=float, default=1e-2)
    parser.add_argument('--sigma', type=float, default=0.8)
    # For pmtl.
    parser.add_argument('--pmtl-warmup-iter', type=int, default=200)
    parser.add_argument('--pmtl-warmup-iter-counter', type=int, default=0)

    parser.add_argument('--update-counter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    args.task_name = 'agg_{}'.format(args.agg) if args.solver == 'agg' else args.solver
    print('{} on {}'.format(args.task_name, args.dataset))

    dataset_dict = {'adult': Adult(split="train"), 'credit': Credit(split="train"), 'compass': Compas(split="train")}
    dataset = dataset_dict[args.dataset]
    setting_dict = {'adult': adult_setting, 'credit': credit_setting, 'compass': compass_setting}
    settings = setting_dict[args.dataset]

    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    obj_arr = from_name(settings['objectives'], dataset.task_names())

    model_arr = [model_from_dataset(args.dataset, dim=dim_dict[args.dataset]) for _ in range(args.n_sub)]
    optimizer_arr = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_arr]
    epoch_loss_pref = []

    # For seperate models, we have N independent models, each with its own optimizer.
    # MOO-SVGD, GradHV do not accept preferences. PMTL accept preferences.
    # if args.solver == 'pmtl':
    # For MOOSVGD and hvgrad, we do not need to specify preferences. However, prefs are remained for visulization purpose.
    pref_mat = torch.Tensor( uniform_pref(number=args.n_sub, clip_eps=0.2) )
    pref_mat_np = pref_mat.detach().cpu().numpy()

    for _ in tqdm(range(args.epoch)):
        loss_batch = []  # shape: (batch, K, n_obj)
        for b, batch in enumerate(trainloader):
            # Construct the loss_mat for each sub-problem.
            loss_mat = []     # shape: (K, n_obj)
            for k in range(args.n_sub):
                logits = model_arr[k](batch)
                batch.update(logits)
                loss_vec = [0] * 2
                for idx, obj in enumerate(obj_arr):
                    loss_vec[idx] = obj(**batch)
                loss_vec = torch.stack(loss_vec)
                loss_vec = normalize_vec(loss_vec, problem=args.dataset)
                loss_mat.append(loss_vec)

            loss_mat = torch.stack(loss_mat)
            assert args.solver in ['hvgrad', 'moosvgd', 'pmtl']
            if args.solver in ['moosvgd', 'pmtl']:
                Jacobian_arr = []
                for pref_idx in range(args.n_sub):
                    gradients, obj_values = calc_gradients(batch, model_arr[pref_idx], obj_arr)
                    Jacobian = torch.stack([flatten_grads(gradients[idx]) for idx in range(2)])
                    Jacobian_arr.append(Jacobian)

            if args.solver == 'hvgrad':
                from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreHVGrad
                solver = CoreHVGrad(args)
                alpha_mat = solver.get_alpha(loss_mat)
            elif args.solver == 'pmtl':
                from libmoon.solver.gradient.methods.bk.core_solver_bk import CorePMTL
                solver = CorePMTL(args, pref_mat)
                is_warmup = args.pmtl_warmup_iter_counter < args.pmtl_warmup_iter
                alpha_mat = solver.get_alpha(Jacobian_arr=Jacobian_arr, loss_mat=loss_mat, is_warmup=is_warmup)
            elif args.solver == 'moosvgd':
                from libmoon.solver.gradient.methods.bk.core_solver_bk import CoreMOOSVGD
                solver = CoreMOOSVGD(args)
                alpha_mat = solver.get_alpha(Jacobian_arr, loss_mat)
                # print()

            for k in range(args.n_sub):
                optimizer_arr[k].zero_grad()
            if type(alpha_mat) == np.ndarray:
                alpha_mat = torch.Tensor(alpha_mat)
            loss = torch.sum( alpha_mat * loss_mat )
            loss.backward()
            for k in range(args.n_sub):
                optimizer_arr[k].step()
                args.update_counter += 1

            loss_batch.append(loss_mat.detach().cpu().numpy())

        loss_batch = np.array(loss_batch)    # shape: (batch, K, n_obj)
        epoch_loss_pref.append( np.mean(loss_batch, axis=0) )   # shape: (epoch, K, n_obj)

    epoch_loss_pref = np.array(epoch_loss_pref)    # shape: (epoch, K, n_obj)
    epoch_loss_pref_final = epoch_loss_pref[-1,:,:] # shape: (K, n_obj)

    import os
    from libmoon.util_global.constant import root_name
    output_folder_name = os.path.join(root_name, 'output', 'mtl', args.task_name, args.dataset, '{}'.format(args.seed))
    os.makedirs(output_folder_name, exist_ok=True)

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


    prefs_np = pref_mat.detach().cpu().numpy()
    prefs_np = prefs_np / np.linalg.norm(prefs_np, axis=1, keepdims=True) * rho_max

    for pref in prefs_np:
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
        for k, v in indicators_dict.items():
            f.write('{}: {}\n'.format(k, v))
        f.write('{}: {}\n'.format('counter', args.update_counter))
    print('Saving indicators to', indicator_name)

    # Finally, save all information into pickle files.
    pickle_name = os.path.join(output_folder_name, 'res.pickle')
    with open(pickle_name, 'wb') as f:
        import pickle

        pickle.dump({
            'args': args,
            'epoch_loss_pref': epoch_loss_pref,
            'epoch_loss_pref_final': epoch_loss_pref_final,
            'pref_mat_np': pref_mat_np,
            'indicators_dict': indicators_dict
        }, f)

    print('Saving all information to', pickle_name)



