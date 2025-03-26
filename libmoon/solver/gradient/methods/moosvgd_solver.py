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


def get_svgd_gradient(Jacobian_array, inputs, losses):
    '''
        :param Jacobian_array.shape: (n_prob, n_obj, n_var)
        :param inputs.shape: (n_prob, n_var)
        :param losses.shape: (n_prob, n_obj)

        Return: gradient: (n_prob, n_obj). Please change the output to (n_prob, n_obj).
    '''
    n_prob = inputs.size(0)
    # G shape (n_prob, n_obj, n_var)
    g_w = [0] * n_prob
    # What is gw?
    for idx in range(n_prob):
        g_w[idx] = torch.Tensor( solve_mgda(Jacobian_array[idx], return_coeff=False) )
    g_w = torch.stack(g_w)  # (n_prob, n_var)
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    kernel = kernel_functional_rbf(losses)
    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs, allow_unused=True)[0]   # (n_prob, n_var)
    gradient = (kernel.mm(g_w) - kernel_grad) / n_prob
    return gradient



def get_svgd_alpha_array(Jacobian_array, loss_array, input):
    '''
        :param Jacobian_array.shape: (n_prob, n_obj, n_var)
        :param losses.shape: (n_prob, n_obj)
        :param inputs.shape: (n_prob, n_var)
        Return: gradient: (n_prob, n_obj). Please change the output to (n_prob, n_obj).
    '''
    # Jacobian_array
    n_prob, n_obj, n_var = Jacobian_array.size()
    alpha_mgda_array = torch.stack([solve_mgda(Jacobian_array[idx]) for idx in range(n_prob)]).double()
    # shape (n_prob, n_obj)
    loss_array_var = Variable(loss_array, requires_grad=True)
    kernel = kernel_functional_rbf(loss_array_var).double()
    # shape (n_prob, n_prob)
    alpha1 = kernel.detach().mm(alpha_mgda_array)
    alpha2 = -0.5 * torch.autograd.grad(kernel.sum(), loss_array_var, allow_unused=True)[0]  # (n_prob, n_var)
    return (alpha1-alpha2) / n_prob
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    # kernel = kernel_functional_rbf(losses)
    # kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), input, allow_unused=True)[0]   # (n_prob, n_var)
    # gradient = (kernel.mm(g_w) - kernel_grad) / n_prob

class MOOSVGDCore():
    def __init__(self, n_var, prefs):
        self.core_name = 'MOOSVGDCore'

    def get_alpha_array(self, Jacobian_arr, losses_arr):
        '''
            Input:
            Jacobian_arr: (n_prob, m, n)
            losses_arr: (n_prob, m)
            Return: (n_prob, m)
        '''
        alpha_array = get_svgd_alpha_array(Jacobian_arr, losses_arr, None)
        return alpha_array

class MOOSVGDSolver(GradBaseSolver):
    def __init__(self, problem, prefs=None, step_size=1e-3, n_epoch=500, tol=1e-3, folder_name=None):
        self.folder_name = folder_name
        self.problem = problem
        self.n_prob = prefs.shape[0]
        self.core_solver = MOOSVGDCore(n_var=problem.n_var, prefs=prefs)
        self.solver_name = 'MOOSVGD'
        self.prefs = prefs
        super().__init__(step_size, n_epoch, tol, self.core_solver)


    def solve(self, x_init):
        return super().solve(self.problem, x_init, self.prefs)


if __name__ == '__main__':
    losses = torch.rand(10, 3)
    kernel = kernel_functional_rbf(losses)





