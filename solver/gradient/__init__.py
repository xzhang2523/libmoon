# from grad_core import solve_mgda_analy

from .base_solver import GradAggSolver
from .mgda_solver import MGDASolver
from .epo_solver import EPOSolver
from .moosvgd import MOOSVGDSolver
from .gradhv import GradHVSolver
from .pmtl import PMTLSolver

from .core_solver import CoreAgg, CoreMGDA, CoreEPO, CoreMOOSVGD


def get_core_solver(mtd, agg_mtd='mtche', pref=None):
    if mtd == 'agg':
        return CoreAgg(pref=pref, agg_mtd=agg_mtd)
    elif mtd == 'mgda':
        return CoreMGDA()
    elif mtd == 'epo':
        return CoreEPO(pref=pref)
    elif mtd == 'moosvgd':
        return CoreMOOSVGD
    else:
        assert False, 'not implemented'