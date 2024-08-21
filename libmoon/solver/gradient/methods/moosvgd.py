# Paper: https://openreview.net/pdf?id=S2-j0ZegyrE
# Paper name: Profiling Pareto Front With Multi-Objective Stein Variational Gradient Descent
from libmoon.solver.gradient.methods.core.mgda_core import solve_mgda
from libmoon.solver.gradient.methods.base_solver import GradBaseSolver
import torch
import math
from torch.autograd import Variable
from torch.optim import SGD
from libmoon.util.constant import solution_eps
from tqdm import tqdm
import numpy as np

def kernel_functional_rbf(losses, bandwidth=5e-6):
    '''
        input losses: (n_prob, n_obj)
        output kernel_matrix: (n_prob, n_prob)
        comments: This function is used to compute the kernel matrix for SVGD.
    '''
    n = losses.shape[0]
    pairwise_distance = torch.norm(losses[:, None] - losses, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    # A = 5e-6  # Noted, this bandwith parameter is important.
    kernel_matrix = torch.exp(-pairwise_distance / bandwidth*h)  # 5e-6 for zdt1,2,3, zxy, Dec 5, 2023
    return kernel_matrix


def median(tensor):
    """
        torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.


# def get_svgd_gradient(Jacobian_array, inputs, losses):
#     '''
#         :param Jacobian_array.shape: (n_prob, n_obj, n_var)
#         :param inputs.shape: (n_prob, n_var)
#         :param losses.shape: (n_prob, n_obj)
#
#         Return: gradient: (n_prob, n_obj). Please change the output to (n_prob, n_obj).
#     '''
#     n_prob = inputs.size(0)
#     # G shape (n_prob, n_obj, n_var)
#     g_w = [0] * n_prob
#     # What is gw?
#     for idx in range(n_prob):
#         g_w[idx] = torch.Tensor( solve_mgda(Jacobian_array[idx], return_coeff=False) )
#     g_w = torch.stack(g_w)  # (n_prob, n_var)
#     # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
#     kernel = kernel_functional_rbf(losses)
#     kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs, allow_unused=True)[0]   # (n_prob, n_var)
#     gradient = (kernel.mm(g_w) - kernel_grad) / n_prob
#     return gradient



def get_svgd_alpha_array(Jacobian_array, loss_array, input):
    '''
        :param Jacobian_array.shape: (n_prob, n_obj, n_var)
        :param losses.shape: (n_prob, n_obj)
        :param inputs.shape: (n_prob, n_var)
        Return: gradient: (n_prob, n_obj). Please change the output to (n_prob, n_obj).
    '''
    # Jacobian_array
    n_prob, n_obj, n_var = Jacobian_array.size()
    alpha_mgda_array = torch.stack([solve_mgda(Jacobian_array[idx]) for idx in range(n_prob)])
    # shape (n_prob, n_obj)

    loss_array_var = Variable(loss_array, requires_grad=True)
    kernel = kernel_functional_rbf(loss_array_var)

    # shape (n_prob, n_prob)
    alpha1 = kernel.detach().mm(alpha_mgda_array)
    alpha2 = -0.5 * torch.autograd.grad(kernel.sum(), loss_array_var, allow_unused=True)[0]  # (n_prob, n_var)

    return (alpha1-alpha2) / n_prob
    # print()





    # print()



    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    # kernel = kernel_functional_rbf(losses)
    # kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), input, allow_unused=True)[0]   # (n_prob, n_var)
    # gradient = (kernel.mm(g_w) - kernel_grad) / n_prob





class MOOSVGDSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)

    def solve(self, problem, x, prefs, args):
        x = Variable(x, requires_grad=True)
        y_arr = []
        optimizer = SGD([x], lr=self.step_size)
        for i in tqdm(range(self.max_iter)):
            y = problem.evaluate(x)
            y_arr.append( y.detach().numpy() )

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
        res['y_arr'] = y_arr
        res['hv_arr'] = [0]
        return res



if __name__ == '__main__':
    losses = torch.rand(10, 3)
    kernel = kernel_functional_rbf(losses)





