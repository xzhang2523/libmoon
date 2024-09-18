from libmoon.util.synthetic import synthetic_init
from libmoon.util.prefs import get_uniform_pref
from libmoon.util.problems import get_problem
from libmoon.solver.gradient.methods import EPOSolver

problem = get_problem(problem_name='ZDT1')
prefs = get_uniform_pref(n_prob=5, n_obj=problem.n_obj, clip_eps=1e-2)
solver = EPOSolver(step_size=1e-2, n_iter=1000, tol=1e-2, problem=problem, prefs=prefs)
res = solver.solve(x=synthetic_init(problem, prefs))
