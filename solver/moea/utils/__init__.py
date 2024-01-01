from solver.moea.utils.decomposition import weighted_sum
from solver.moea.utils.decomposition import Tchebycheff, modified_Tchebycheff

from solver.moea.utils.decomposition import penalty_boundary_intersection
from solver.moea.utils.utils_ea import dominance_min

def get_decomposition(name):

    decom_methods = {
        "ws": weighted_sum,
        "tch": Tchebycheff,
        "mtch": modified_Tchebycheff,
        "pbi": penalty_boundary_intersection,
    }

    if name not in decom_methods:
        raise Exception("Decomposition method not found.")

    return decom_methods[name]