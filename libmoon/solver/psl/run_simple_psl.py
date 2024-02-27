# from problem.sy
from solver.psl.model import SimplePSLModel
from libmoon.util_global.constant import problem_dict, FONT_SIZE
import argparse
from tqdm import tqdm
import numpy as np
import torch
from libmoon.util_global.scalarization import tche
from libmoon.util_global.weight_factor.funs import uniform_pref



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='zdt1', type=str)
    parser.add_argument('--n_var', default=30, type=int)
    parser.add_argument('--dist', default='dirichlet', type=str )
    parser.add_argument('--epoch', default=100, type=int )
    parser.add_argument('--batch-size', default=128, type=int )
    parser.add_argument('--lr', default=1e-3, type=float )


    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('cuda is available')
    else:
        device = torch.device("cpu")
        print('cuda is not available')

    args.device = device
    problem = problem_dict[args.problem]
    model = SimplePSLModel(problem, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    loss_history = []
    for _ in tqdm(range(args.epoch)):
        prefs = torch.Tensor( np.random.dirichlet(np.ones(problem.n_obj), args.batch_size) ).to(args.device)

        xs = model(prefs)
        fs = problem.evaluate(xs)
        g = tche(fs, prefs)
        loss = torch.mean(g)
        loss_history.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    from matplotlib import pyplot as plt

    test_pref = torch.Tensor( uniform_pref(20, problem.n_obj) ).to(args.device)
    predict_x = model(test_pref)
    predict_y = problem.evaluate(predict_x)
    predict_y_np = predict_y.cpu().detach().numpy()

    plt.scatter(predict_y_np[:, 0], predict_y_np[:, 1], marker='o', color='r', facecolors='none', label='predict')

    plt.xlabel('$L_1$', fontsize=FONT_SIZE)
    plt.ylabel('$L_2$', fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)


    plt.show()