import torch

from solver.gradient.base_solver import GradBaseSolver

from autograd import grad
from matplotlib import pyplot as plt
from solver.gradient.min_norm_solvers_numpy import MinNormSolver
import numpy as np

from util.constant import problem_dict
from torch.autograd import Variable

from tqdm import tqdm
from util.constant import solution_eps


problem = problem_dict['vlmop2']



def get_d_moomtl(grads):
    """
        calculate the gradient direction for MOO-MTL
    """
    nobj, dim = grads.shape
    sol, nd = MinNormSolver.find_min_norm_element(grads)
    return sol


def get_d_paretomtl(grads, value, weights, i):
    # calculate the gradient direction for Pareto MTL
    nobj, dim = grads.shape

    # check active constraints
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis=1,
                                                                               keepdims=True)
    w = normalized_rest_weights - normalized_current_weight

    # solve QP
    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0

    vec = np.concatenate((grads, np.dot(w[idx], grads)), axis=0)
    # use MinNormSolver to solve QP
    sol, nd = MinNormSolver.find_min_norm_element(vec)


    # reformulate ParetoMTL as linear scalarization method, return the weights
    weight0 = sol[0] + np.sum(np.array([sol[j] * w[idx][j - 2, 0] for j in np.arange(2, 2 + np.sum(idx))]))
    weight1 = sol[1] + np.sum(np.array([sol[j] * w[idx][j - 2, 1] for j in np.arange(2, 2 + np.sum(idx))]))
    weight = np.stack([weight0, weight1])

    return weight




def get_d_paretomtl_init(grads, value, weights, i):
    # calculate the gradient direction for Pareto MTL initialization
    nobj, dim = grads.shape

    # check active constraints
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis=1,
                                                                               keepdims=True)
    w = normalized_rest_weights - normalized_current_weight

    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0

    if np.sum(idx) <= 0:
        return np.zeros(nobj)
    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vec = np.dot(w[idx], grads)
        sol, nd = MinNormSolver.find_min_norm_element(vec)

    # calculate the weights
    weight0 = np.sum(np.array([sol[j] * w[idx][j, 0] for j in np.arange(0, np.sum(idx))]))
    weight1 = np.sum(np.array([sol[j] * w[idx][j, 1] for j in np.arange(0, np.sum(idx))]))
    weight = np.stack([weight0, weight1])

    return weight


def circle_points(r, n):
    # generate evenly distributed preference vector
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles





def pareto_mtl_search(ref_vecs, i, t_iter=100, n_dim=20, step_size=1):
    """
        Pareto MTL.
    """
    # randomly generate one solution
    x = np.random.uniform(-0.5, 0.5, problem.n_var)
    x = Variable(torch.Tensor(x).unsqueeze(0), requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=step_size)

    # find the initial solution
    for t in range( int(t_iter * 0.2) ):

        # f, f_dx = concave_fun_eval(x)
        f = problem.evaluate( x )
        f_dx = [0] * problem.n_obj
        for obj_idx in range(problem.n_obj):
            f[0][obj_idx].backward(retain_graph=True)
            f_dx[obj_idx] = x.grad.clone()
            x.grad.zero_()

        f_dx = torch.stack(f_dx)
        f_dx_np = f_dx.detach().numpy().squeeze()
        f_np = f.detach().numpy().squeeze()

        weights = get_d_paretomtl_init(f_dx_np, f_np, ref_vecs, i)

        optimizer.zero_grad()
        torch.sum(torch.tensor(weights).unsqueeze(0) * f).backward()
        optimizer.step()

        # print()




    # find the Pareto optimal solution
    for t in range(int(t_iter * 0.8)) :
        f = problem.evaluate(x)
        f_dx = [0] * problem.n_obj
        for obj_idx in range(problem.n_obj):
            f[0][obj_idx].backward(retain_graph=True)
            f_dx[obj_idx] = x.grad.clone()
            x.grad.zero_()

        f_dx = torch.stack(f_dx)
        f_dx_np = f_dx.detach().numpy().squeeze()
        f_np = f.detach().numpy().squeeze()

        weights = get_d_paretomtl(f_dx_np, f_np, ref_vecs, i)
        optimizer.zero_grad()
        torch.sum(torch.tensor(weights).unsqueeze(0) * f).backward()
        optimizer.step()

    return x, f


def run(method='ParetoMTL', num=10):
    """
        run method on the synthetic example
        method: optimization method {'ParetoMTL', 'MOOMTL', 'Linear'}
        num: number of solutions
    """

    # pf = create_pf()
    pf = problem.get_pf()

    f_value_list = []
    weights = circle_points([1], [num])[0]

    # weigths.shape: (n_prob, n_obj)


    for i in tqdm(range(num)):
        x, f = pareto_mtl_search(ref_vecs=weights, i=i, t_iter=3000, step_size=1e-2)
        f_np = f.detach().numpy().flatten()
        f_value_list.append(f_np)


    f_value = np.array(f_value_list)
    plt.plot(pf[:, 0], pf[:, 1])
    plt.scatter(f_value[:, 0], f_value[:, 1], c='r', s=80)

    plt.show()






if __name__ == '__main__':
    run('ParetoMTL')









class PMTLSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)



    def solve(self, problem, x, prefs, args):
        if args.n_obj != 2:
            assert False, 'hvgrad only supports 2 obj problem'

        x = Variable(x, requires_grad=True)
        warmup_iter = self.max_iter // 5
        optimizer = torch.optim.SGD([x], lr=self.step_size)


        for iter_idx in tqdm( range(self.max_iter) ):
            y = problem.evaluate(x)
            y_np = y.detach().numpy()

            grad_arr = [0] * args.n_prob
            for prob_idx in range(args.n_prob):
                grad_arr[prob_idx] = [0] * args.n_obj
                for obj_idx in range(args.n_obj):
                    y[prob_idx][obj_idx].backward(retain_graph=True)
                    grad_arr[prob_idx][obj_idx] = x.grad[prob_idx].clone()
                    # grad_arr: (n_prob, n_obj, n_var)
                    x.grad.zero_()
                grad_arr[prob_idx] = torch.stack(grad_arr[prob_idx])


            grad_arr = torch.stack(grad_arr)
            grad_arr_np = grad_arr.detach().numpy()
            if iter_idx < warmup_iter:
                weights = [get_d_paretomtl_init(grad_arr_np[i], y_np[i], prefs, i) for i in range(args.n_prob)]
            else:
                weights = [get_d_paretomtl(grad_arr_np[i], y_np[i], prefs, i) for i in range(args.n_prob)]

            # print()
            optimizer.zero_grad()
            torch.sum(torch.tensor(weights) * y).backward()
            optimizer.step()

            if 'lb' in dir(problem):
                x.data = torch.clamp(x.data, problem.lb + solution_eps, problem.ub - solution_eps)

        res={}
        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = [0]
        return res