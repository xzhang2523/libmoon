from .base_solver import GradAggSolver
from .mgda_solver import MGDASolver
from .epo_solver import EPOSolver
from .moosvgd import MOOSVGDSolver
from .gradhv import GradHVSolver
from .pmtl import PMTLSolver



from .core_solver import CoreAgg, CoreMGDA, CoreEPO, CoreMOOSVGD, CoreHVGrad

def get_core_solver(args, pref=None):
    if args.solver == 'agg':
        return CoreAgg(pref=pref, agg_mtd=args.agg_mtd)
    elif args.solver == 'mgda':
        return CoreMGDA()
    elif args.solver == 'epo':
        return CoreEPO(pref=pref)
    elif args.solver == 'moosvgd':
        return CoreMOOSVGD(args=args)
    elif args.solver == 'hvgrad':
        return CoreHVGrad(args=args)
    else:
        assert False, 'not implemented'