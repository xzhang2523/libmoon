from libmoon.solver.psl.model import SimplePSLModel

from libmoon.util_global.constant import get_problem, FONT_SIZE, agg_dict

import argparse
from tqdm import tqdm
import numpy as np
import torch
from libmoon.util_global.weight_factor import uniform_pref
from matplotlib import pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from libmoon.fig_util.plot import get_psl_info
from torch.autograd import Variable
from libmoon.util_global.zero_order import ES_gradient_estimation_batch
from libmoon.solver.gradient.methods.epo_solver import solve_epo, EPO_LP

from libmoon.solver.gradient.methods.pmgda_solver import solve_pmgda, constraint, get_Jhf
from libmoon.util_global.constant import root_name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='ZDT1', type=str)
    parser.add_argument('--agg', default='softtche', type=str)
    parser.add_argument('--solver', default='agg', type=str)     # ['agg', 'epo','moosvgd', ...]
    parser.add_argument('--solver-ec', default='na', type=str)

    parser.add_argument('--n_var', default=10, type=int)
    parser.add_argument('--dist', default='dirichlet', type=str)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # For pmgda
    parser.add_argument('--h-eps', type=float, default=1e-2)
    parser.add_argument('--sigma', type=float, default=0.8)


    args = parser.parse_args()
    if args.solver == 'agg':
        if args.solver_ec == 'es':
            args.task_name = 'psl_agg_{}_es'.format(args.agg)
        else:
            args.task_name = 'psl_agg_{}'.format(args.agg)
    else:
        args.task_name = 'psl_{}'.format(args.solver)


    print('task name: {} on {}'.format(args.task_name, args.problem))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('cuda is available')
    else:
        device = torch.device("cpu")
        print('cuda is not available')


    args.device = device
    problem = get_problem(args.problem, args.n_var)
    args.n_obj = problem.n_obj

    model = SimplePSLModel(problem, args).to(args.device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_history = []

    for _ in tqdm(range(args.epoch)):
        prefs = torch.Tensor( np.random.dirichlet(np.ones(problem.n_obj), args.batch_size) ).to(args.device)
        xs = model(prefs)
        fs = problem.evaluate(xs)
        if args.solver == 'agg':
            if args.solver_ec == 'es':
                # Part 1. Estimating term A.

                agg_func = agg_dict[args.agg]
                fs_var = Variable(fs, requires_grad=True)
                g = agg_func(fs_var, prefs)
                loss_g = torch.mean(g)
                loss_history.append(loss_g.cpu().detach().numpy())
                g.sum().backward()
                termA = (fs_var.grad).unsqueeze(1)
                # Part 2. Estimating term B.

                termB = ES_gradient_estimation_batch(problem, xs.cpu().detach().numpy())
                termB = torch.Tensor(termB).to(args.device)
                xs = model(prefs)
                res = termA @ termB
                loss = torch.mean(res @ xs.unsqueeze(2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                agg_func = agg_dict[args.agg]
                g = agg_func(fs, prefs)
                loss = torch.mean(g)
                loss_history.append(loss.cpu().detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif args.solver == 'epo':
            epo_arr = [EPO_LP(m=args.n_obj, n=args.n_var, r=np.array(1 / pref.cpu())) for pref in prefs]
            alpha_arr = [0] * args.batch_size
            for prob_idx in range(args.batch_size):
                Jacobian = torch.autograd.functional.jacobian(lambda x: problem.evaluate(x).squeeze(), xs[prob_idx])
                Jacobian = torch.squeeze(Jacobian)
                _, alpha = solve_epo(Jacobian, losses=fs[prob_idx], pref=prefs[prob_idx], epo_lp=epo_arr[prob_idx])
                alpha_arr[prob_idx] = alpha
            alpha_arr = torch.Tensor(np.array(alpha_arr)).to(fs.device)
            xs = model(prefs)
            fs = problem.evaluate(xs)
            loss = torch.mean( torch.sum(alpha_arr * fs, axis=1) )
            loss_history.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif args.solver == 'pmgda':
            xt = Variable(xs, requires_grad=True)
            yt = problem.evaluate(xt)
            alpha_arr = [0] * args.batch_size
            for prob_idx in range(args.batch_size):
                Jacobian = torch.autograd.functional.jacobian(lambda ph: problem.evaluate(ph).squeeze(),
                                                              xs[prob_idx].unsqueeze(0))
                Jacobian = torch.squeeze(Jacobian)
                pref = prefs[prob_idx]
                # (Step 2). Get the gradient of the constraint.
                h = constraint(yt[prob_idx].unsqueeze(0), pref=pref, args=args)
                h.backward(retain_graph=True)
                grad_h = xt.grad[prob_idx].detach().cpu().clone()
                h_val = float(h.detach().cpu().clone().numpy())
                Jhf = get_Jhf(fs[prob_idx], pref, args)
                # replace it to mgda loss
                _, alpha = solve_pmgda(Jacobian, grad_h, h_val, args, return_coeff=True,
                                       Jhf=Jhf)  # combine the gradient information
                alpha_arr[prob_idx] = alpha

            alpha_arr = torch.Tensor(np.array(alpha_arr)).to(fs.device)
            xs = model(prefs)
            fs = problem.evaluate(xs)
            loss = torch.mean(torch.sum(alpha_arr * fs, axis=1))
            loss_history.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            assert False, 'Not supported solver'

    folder_name = os.path.join(root_name, 'output', 'psl', args.task_name, args.problem)
    os.makedirs(folder_name, exist_ok=True)

    fig = plt.figure()
    plt.plot( loss_history )
    plt.xlabel('Epoch', fontsize=FONT_SIZE)
    plt.ylabel('PSL Loss', fontsize=FONT_SIZE)
    loss_name = os.path.join(folder_name, 'learning_curve.pdf')
    print('save to {}'.format(loss_name))
    plt.savefig( loss_name)

    fig = plt.figure()
    test_pref = torch.Tensor( uniform_pref(20, problem.n_obj) ).to(args.device)
    predict_x = model(test_pref)
    predict_y = problem.evaluate(predict_x)
    predict_y_np = predict_y.cpu().detach().numpy()

    pref, x, y = get_psl_info(args, model, problem)
    plt.scatter(y[:, 0], y[:, 1], marker='o', color='tomato', facecolors='none', label='Predict')

    plt.scatter(pref[:, 0], pref[:, 1], marker='o', color='skyblue', label='Pref')
    plt.plot([0,1], [1,0], color='skyblue')

    for idx, (pp, yy) in enumerate(zip(pref, y)):
        if idx ==0:
            plt.plot([pp[0], yy[0]], [pp[1], yy[1]], color='skyblue', label='Mapping', linestyle ='dashed')
        else:
            plt.plot([pp[0], yy[0]], [pp[1], yy[1]], color='skyblue', linestyle ='dashed')

    pf = problem.get_pf()

    plt.xlabel('$L_1$', fontsize=FONT_SIZE)
    plt.ylabel('$L_2$', fontsize=FONT_SIZE)
    plt.plot(pf[:, 0], pf[:, 1], color='tomato', label='PF')
    plt.legend(fontsize=16)
    fig_name = os.path.join(folder_name, 'psl.pdf')
    plt.savefig(fig_name)
    print('save to {}'.format(fig_name))