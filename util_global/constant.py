from util_global.scalarization import ls, mtche, tche, pbi, cosmos
from problem.synthetic import VLMOP1, VLMOP2, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from problem.synthetic import MAF1
import os
from numpy import array
import torch



FONT_SIZE = 20
solution_eps = 1e-5
scalar_dict = {
    'ls' : ls,
    'mtche' : mtche,
    'tche' : tche,
    'pbi' : pbi,
    'cosmos' : cosmos
}

problem_dict = {
    'zdt1': ZDT1(),
    'zdt2': ZDT2(),
    'zdt3': ZDT3(),
    'zdt4': ZDT4(),
    'zdt6': ZDT6(),
    'vlmop1': VLMOP1(),
    'vlmop2': VLMOP2(),
    'maf1': MAF1()
}


hv_ref_dict = {
    'vlmop1': array([1.0, 1.0]),
    'vlmop2': array([4.0, 4.0]),
    'maf1': array([2.0, 2.0, 2.0]),
    'mnist': array([3.0, 3.0])
}

def get_hv_ref_dict(problem_name):
    if problem_name.startswith('zdt'):
        ref = array([1.0, 1.0])
    else:
        ref = hv_ref_dict[problem_name]
    return ref + 0.5

root_name = os.path.dirname(os.path.dirname(__file__))
def is_pref_based(mtd):
    if mtd in ['epo', 'mgda', 'agg', 'pmgda']:
        return True
    else:
        return False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('cuda is available')
    else:
        device = torch.device("cpu")
        print('cuda is not available')
    return device

def get_param_num(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)