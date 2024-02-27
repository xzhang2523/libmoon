from torch.autograd import Variable
from torch.optim import SGD
from torch import Tensor
from ...util_global.constant import scalar_dict, solution_eps, get_hv_ref_dict
import torch
from tqdm import tqdm
from pymoo.indicators.hv import HV


class GradBaseSolver:
    def __init__(self, step_size, max_iter, tol):
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem, x, prefs, args):
        '''
            :param problem:
            :param x:
            :param agg:
            :return:
                is a dict with keys: x, y
        '''

        # The abstract class cannot be implemented directly.
        raise NotImplementedError



class GradAggSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol):
        super().__init__(step_size, max_iter, tol)

    def solve(self, problem, x, prefs, args):
        x = Variable(x, requires_grad=True)

        # ref_point = array([2.0, 2.0])
        ind = HV(ref_point = get_hv_ref_dict(args.problem_name))


        hv_arr = []
        y_arr = []

        prefs = Tensor(prefs)
        optimizer = SGD([x], lr=self.step_size)
        agg_func = scalar_dict[args.agg]
        res = {}
        for i in tqdm(range(self.max_iter)):

            y = problem.evaluate(x)
            hv_arr.append(ind.do(y.detach().numpy()))

            agg_val = agg_func(y, prefs)
            optimizer.zero_grad()
            torch.sum(agg_val).backward()
            optimizer.step()

            y_arr.append(y.detach().numpy())

            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, torch.Tensor(problem.lbound) + solution_eps, torch.Tensor(problem.ubound)-solution_eps)

        res['x'] = x.detach().numpy()
        res['y'] = y.detach().numpy()
        res['hv_arr'] = hv_arr
        res['y_arr'] = y_arr
        return res