# from grad_core import solve_mgda_analy

from .base_solver import GradAggSolver
from .mgda_solver import MGDASolver
from .epo_solver import EPOSolver
from .moosvgd import MOOSVGDSolver
from .gradhv import GradHVSolver
from .pmtl import PMTLSolver

from .core_solver import CoreAgg


def get_core_solver(mtd):
    if mtd == 'agg':
        return CoreAgg(agg_mtd='ls')
    else:
        assert False, 'not implemented'