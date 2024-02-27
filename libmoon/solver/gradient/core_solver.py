import numpy as np
from .mgda_core import solve_mgda
from .epo_solver import EPO_LP
import torch
from .gradhv import HvMaximization
from ...util_global.constant import get_hv_ref_dict



class CoreHVGrad:
    def __init__(self, args):
        self.args = args
        self.hv_solver = HvMaximization(args.n_prob, args.n_obj, get_hv_ref_dict(args.problem))

    def get_alpha(self, losses):
        alpha = self.hv_solver.compute_weights(losses).T
        return alpha



class CoreMOOSVGD:
    def __init__(self):
        pass

    def get_alpha(self):
        return 0


class CoreMGDA:
    def __init__(self):
        pass

    def get_alpha(self, G, losses=None, pref=None):
        _, alpha = solve_mgda(G, return_coeff=True)
        return alpha


class CoreGrad:
    def __init__(self):
        pass



class CoreEPO(CoreGrad):
    def __init__(self, pref):
        self.pref = pref
        self.epo_lp = EPO_LP(m=len(pref), n=1, r=1/np.array(pref))


    def get_alpha(self, G, losses):
        if type(G) == torch.Tensor:
            G = G.detach().cpu().numpy().copy()
        GG = G @ G.T

        alpha = self.epo_lp.get_alpha(losses, G=GG, C=True)
        return alpha




class CoreAgg(CoreGrad):
    def __init__(self, pref, agg_mtd='ls'):
        self.agg_mtd = agg_mtd
        self.pref = pref

    def get_alpha(self, G, losses):
        if self.agg_mtd == 'ls':
            alpha = self.pref
        elif self.agg_mtd == 'mtche':
            idx = np.argmax(losses)
            alpha = np.zeros_like(self.pref )
            alpha[idx] = 1.0
        else:
            assert False
        return alpha



if __name__ == '__main__':
    agg = CoreAgg( pref=np.array([1, 0]) )


