from solver.ea.utils.decomposition import weighted_sum
from solver.ea.utils.decomposition import Tchebycheff
from solver.ea.utils.decomposition import penalty_boundary_intersection
from solver.ea.utils.utils_ea import dominance_min

def get_decomposition(name):

    decom_methods = {
        "ws": weighted_sum,
        "tch": Tchebycheff,
        "pbi": penalty_boundary_intersection,
    }

    if name not in decom_methods:
        raise Exception("Decomposition method not found.")

    return decom_methods[name]