
from libmoon.solver.psl.core_psl import AggPSLSolver
from libmoon.util_global import get_problem, uniform_pref
from torch import Tensor


if __name__ == '__main__':

    agg = 'ls'
    problem = get_problem(problem_name='ZDT1', n_var=10)
    solver = AggPSLSolver(problem, batch_size=128, device='cuda', lr=1e-4, epoch=1000, agg=agg, use_es=False)
    model = solver.solve()

    prefs = uniform_pref(n_prob=100, n_obj=problem.n_obj, clip_eps=1e-2)
    eval_y_np = problem.evaluate(model(Tensor(prefs).cuda())).cpu().detach().numpy()




