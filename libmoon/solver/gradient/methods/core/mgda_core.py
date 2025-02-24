import torch
from numpy import array
import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def solve_mgda_analy(grad_1, grad_2):
    '''
        Solve_mgda_analy only support 2-objective case.
        grad_i.shape: (n,).
        This function support grad_i as both Tensor and numpy.
    '''
    v1v1 = grad_1 @ grad_1
    v2v2 = grad_2 @ grad_2
    v1v2 = grad_1 @ grad_2

    if v1v2 >= v1v1:
        gamma = 0.999
    elif v1v2 >= v2v2:
        gamma = 0.001
    else:
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
    coeff = torch.Tensor([gamma, 1 - gamma] )
    # gw = coeff[0] * grad_1 + coeff[1] * grad_2
    # else:
    #     return gw
    return coeff





def solve_mgda(Jacobian):
    '''
        Input Jacobian: (m,n).
        Output alpha: (m,)
    '''

    # m : n_obj
    # n : n_var

    m = Jacobian.shape[0]
    if m == 2:
        return solve_mgda_analy(Jacobian[0], Jacobian[1])
    else:
        Q = (Jacobian @ Jacobian.T).cpu().detach().numpy()

        Q = matrix(np.float64(Q))
        p = np.zeros(m)
        A = np.ones(m)

        A = matrix(A, (1, m))
        b = matrix(1.0)

        G_cvx = -np.eye(m)
        h = [0.0] * m
        h = matrix(h)

        G_cvx = matrix(G_cvx)
        p = matrix(p)
        sol = solvers.qp(Q, p, G_cvx, h, A, b)

        res = np.array(sol['x']).squeeze()
        alpha = res / sum(res)  # important. Does res already satisfy sum=1?
        return alpha


