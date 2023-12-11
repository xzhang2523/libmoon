import torch
from numpy import array
import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def solve_mgda_analy(grad_1, grad_2, return_coeff = False):
    '''
        Noted that, solve_mgda_analy only support 2-objective case.
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

    coeff = array([gamma, 1-gamma])
    gw = coeff[0] * grad_1 + coeff[1] * grad_2
    if return_coeff:
        return gw, coeff
    else:
        return gw


def solve_mgda(G, return_coeff=False):
    '''
        input G: (m,n).
        output gw (n,).
        comments: This function is used to solve the dual MGDA problem. It can handle m>2.
    '''
    if type(G) == torch.Tensor:
        G = G.detach().cpu().numpy().copy()


    m = G.shape[0]
    if m == 2:
        return solve_mgda_analy(G[0], G[1], return_coeff=return_coeff)
    else:
        Q = G @ G.T
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
        res = res / sum(res)  # important
        gw = torch.Tensor( res @ G )

        if return_coeff:
            return gw, res
        else:
            return gw


