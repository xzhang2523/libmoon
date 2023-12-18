import numpy as np 
from numpy import linalg as LA


def weighted_sum(f: np.ndarray, 
                 lamda: np.ndarray
                 ) -> np.ndarray: 

    assert f.ndim * lamda.ndim == 1 
    return np.sum(f * lamda)

def Tchebycheff(f: np.ndarray, 
                lamda: np.ndarray, 
                ref_point: np.ndarray, 
                ) -> np.ndarray: 
    
    assert f.ndim * lamda.ndim * ref_point.ndim == 1 
    return np.max(lamda * np.abs(f-ref_point))

def penalty_boundary_intersection(f: np.ndarray, 
                                  lamda: np.ndarray, 
                                  ref_point: np.ndarray, 
                                  theta: float, 
                                  ) -> np.ndarray: 
    
    assert f.ndim * lamda.ndim * ref_point.ndim == 1 
    d1 = np.dot(f-ref_point, lamda) / LA.norm(lamda) 
    d2 = LA.norm(f - (ref_point+d1*lamda))
    return d1 + theta * d2