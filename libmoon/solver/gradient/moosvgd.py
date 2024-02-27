from .mgda_core import solve_mgda
from .base_solver import GradBaseSolver

import torch
import math
from torch.autograd import Variable
from torch.optim import SGD
import sys
from ...util_global.constant import solution_eps
from tqdm import tqdm

def kernel_functional_rbf(losses):
    '''
        input losses: (n_prob, n_obj)
        output kernel_matrix: (n_prob, n_prob)
        comments: This function is used to compute the kernel matrix for SVGD.
    '''
    n = losses.shape[0]
    # losses shape : (10,) * (3,)
    pairwise_distance = torch.norm(losses[:, None] - losses, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    A = 5e-6  # Noted, this bandwith parameter is important.
    kernel_matrix = torch.exp(-pairwise_distance / A*h)  # 5e-6 for zdt1,2,3, zxy, Dec 5, 2023
    return kernel_matrix

def median(tensor):
    """
        torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

def get_svgd_gradient(G, inputs, losses):
    '''
    :param G.shape: (n_prob, n_obj, n_var)
    :param inputs.shape: (n_prob, n_var)
    :param losses.shape: (n_prob, n_obj)
    :return:
    '''
    n_prob = inputs.size(0)
    # G shape (n_prob, n_obj, n_var)
    g_w = [0] * n_prob

    for idx in range(n_prob):
        g_w[idx] = torch.Tensor( solve_mgda(G[idx], return_coeff=False) )

    g_w = torch.stack(g_w)  # (n_prob, n_var)
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5

    kernel = kernel_functional_rbf(losses)
    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs, allow_unused=True)[0]   # (n_prob, n_var)
    gradient = (kernel.mm(g_w) - kernel_grad) / n_prob

    return gradient



class MOOSVGDSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)

    def solve(self, problem, x, prefs, args):
        x = Variable(x, requires_grad=True)
        optimizer = SGD([x], lr=self.step_size)
        for i in tqdm(range(self.max_iter)):
            y = problem.evaluate(x)
            grad_arr = [0] * args.n_prob
            for prob_idx in range(args.n_prob):
                grad_arr[prob_idx] = [0] * args.n_obj
                for obj_idx in range(args.n_obj):
                    y[prob_idx][obj_idx].backward(retain_graph=True)
                    grad_arr[prob_idx][obj_idx] = x.grad[prob_idx].clone()
                    x.grad.zero_()
                grad_arr[prob_idx] = torch.stack(grad_arr[prob_idx])

            grad_arr = torch.stack(grad_arr).detach()
            gw = get_svgd_gradient(grad_arr, x, y)
            optimizer.zero_grad()
            x.grad = gw
            optimizer.step()

            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound) - solution_eps)

        res={}
        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = [0]
        return res



if __name__ == '__main__':
    losses = torch.rand(10, 3)
    kernel = kernel_functional_rbf(losses)





