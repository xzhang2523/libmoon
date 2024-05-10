import matplotlib.pyplot as plt
from torch.utils import data
from libmoon.problem.mtl.loaders import ADULT
from libmoon.problem.mtl.objectives import from_name
from libmoon.problem.mtl.model_utils import model_from_dataset, dim_dict
from libmoon.problem.mtl.settings import adult_setting, credit_setting, compass_setting
# To achieve discrete solutions.
import argparse
import torch
import numpy as np
from tqdm import tqdm
from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global.constant import agg_dict, color_arr
from libmoon.util_global.grad_util import calc_gradients, flatten_grads




if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA is available")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--solver', type=str, default='pmgda')

    parser.add_argument('--agg', type=str, default='ls')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-sub', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n-obj', type=int, default=2)



    # For pmgda
    parser.add_argument('--h-eps', type=float, default=1e-2)
    parser.add_argument('--sigma', type=float, default=0.8)


    args = parser.parse_args()
    if args.solver == 'agg':
        args.task_name = 'agg_{}'.format(args.agg)
    else:
        args.task_name = args.solver

    print('{} on {}'.format(args.task_name, args.dataset))
    dataset = ADULT(split="train")
    settings = adult_setting
    trainloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    obj_arr = from_name(settings['objectives'], dataset.task_names())

    model_arr = [model_from_dataset(args.dataset, dim=dim_dict[args.dataset]) for _ in range(args.n_sub)]
    optimizer_arr = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_arr]
    prefs = torch.Tensor( uniform_pref(number=args.n_sub, clip_eps=0.2) )
    epoch_loss_pref = []


    # For seperate models, we have N independent models, each with its own optimizer.
    for _ in tqdm(range(args.epoch)):
        loss_batch = []
        for b, batch in enumerate(trainloader):
            loss_pref_arr = [0] * len(prefs)
            for pref_idx, pref in enumerate(prefs):
                # data.keys() : dict_keys(['data', 'labels', 'sensible_attribute'])
                logits = model_arr[pref_idx](batch)
                batch.update(logits)
                loss_vec = [0] * 2
                for idx, obj in enumerate(obj_arr) :
                    if idx==0:
                        loss = obj( **batch )/5
                    else:
                        loss = obj( **batch )
                    loss_vec[idx] = loss
                loss_vec = torch.stack(loss_vec)

                # Calc the gradient, and get the alpha.
                if args.solver != 'agg':
                    gradients, obj_values = calc_gradients(batch, model_arr[pref_idx], obj_arr)
                    Jacobian = torch.stack([flatten_grads(gradients[idx]) for idx in range(2)])

                    if args.solver == 'epo':
                        from libmoon.solver.gradient.methods.core_solver import CoreEPO
                        core_epo = CoreEPO(pref)
                        alpha = torch.Tensor( core_epo.get_alpha(Jacobian, loss_vec) )
                    elif args.solver == 'mgda':
                        from libmoon.solver.gradient.methods.core_solver import CoreMGDA
                        core_mgda = CoreMGDA()
                        alpha = torch.Tensor( core_mgda.get_alpha(Jacobian) )
                    elif args.solver == 'pmgda':
                        from libmoon.solver.gradient.methods.core_solver import CorePMGDA
                        from libmoon.solver.gradient.methods.pmgda_core import get_nn_pmgda_componets
                        h_val, Jhf = get_nn_pmgda_componets(loss_vec, pref, args)
                        grad_h = torch.Tensor(Jhf) @ Jacobian
                        core_pmgda = CorePMGDA(args)
                        alpha = core_pmgda.get_alpha(Jacobian, grad_h, h_val, args, return_coeff=True, Jhf=Jhf)
                        alpha = torch.Tensor( alpha )
                if args.solver == 'agg':
                    agg_func = agg_dict[args.agg]
                    scalar_loss = torch.squeeze(agg_func(loss_vec.unsqueeze(0), pref.unsqueeze(0)))
                else:
                    scalar_loss = torch.dot(loss_vec, alpha)

                optimizer_arr[pref_idx].zero_grad()
                scalar_loss.backward()
                optimizer_arr[pref_idx].step()
                loss_pref_arr[pref_idx] = loss_vec.detach().cpu().numpy()
            loss_batch.append(np.array(loss_pref_arr))
        loss_batch = np.array(loss_batch)
        epoch_loss_pref.append( np.mean(loss_batch, axis=0) )



    epoch_loss_pref = np.array(epoch_loss_pref)
    epoch_loss_pref_final = epoch_loss_pref[-1,:,:]

    import os
    from libmoon.util_global.constant import root_name
    output_folder_name = os.path.join(root_name, 'output', 'mtl', args.task_name, 'adult')
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
    prefs_np = prefs.detach().cpu().numpy()
    prefs_np = prefs_np / np.linalg.norm(prefs_np, axis=1, keepdims=True) * rho_max

    for pref in prefs_np:
        plt.plot([0, pref[0]], [0, pref[1]], color='grey', linestyle='--')

    plt.xlabel('$L_1$')
    plt.ylabel('$L_2$')

    fig_name = os.path.join(output_folder_name, 'res.pdf')
    print('Saving figure to', fig_name)
    plt.savefig(fig_name)



