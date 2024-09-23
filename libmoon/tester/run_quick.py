from libmoon.util.synthetic import synthetic_init
from libmoon.util.prefs import get_uniform_pref
from libmoon.util.problems import get_problem
from libmoon.solver.gradient.methods import EPOSolver

# problem = get_problem(problem_name='ZDT1')
# prefs = get_uniform_pref(n_prob=5, n_obj=problem.n_obj, clip_eps=1e-2)
# solver = EPOSolver(step_size=1e-2, n_iter=1000, tol=1e-2, problem=problem, prefs=prefs)
# res = solver.solve(x=synthetic_init(problem, prefs))

from libmoon.solver.psl.core_psl import BasePSLSolver
from libmoon.util import get_problem
from libmoon.util.prefs import get_uniform_pref
from torch import Tensor

problem = get_problem(problem_name='ZDT1')
# agg list [ ’ls ’, ’tche ’, ’mtche ’, ’pbi ’, ... ]
prefs = get_uniform_pref(n_prob=100, n_obj=problem.n_obj, clip_eps=1e-2)
solver = BasePSLSolver(problem, solver_name='agg_ls')
model, _ = solver.solve()
eval_y = problem.evaluate(model(Tensor(prefs).cuda()))
