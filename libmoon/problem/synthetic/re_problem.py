import numpy as np
import torch
from torch import Tensor
from libmoon.problem.synthetic.mop import BaseMOP
from numpy import array


class RE21(BaseMOP):
    def __init__(self, n_var=4, n_obj=2, lbound=np.zeros(4), ubound=np.ones(4)):
        self.problem_name = 'RE21'
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_cons = 0
        self.n_original_constraints = 0
        self.ideal = array([1237.8414230005742, 0.002761423749158419])
        self.nadir = np.array([2886.3695604236013, 0.039999999999998245])
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma
        self.ubound = np.full(self.n_var, 3 * tmp_val)
        self.lbound = np.zeros(self.n_var)
        self.lbound[0] = tmp_val
        self.lbound[1] = np.sqrt(2.0) * tmp_val
        self.lbound[2] = np.sqrt(2.0) * tmp_val
        self.lbound[3] = tmp_val

    def _evaluate_numpy(self, x):
        n_sub = len(x)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        f = np.zeros((n_sub, self.n_obj) )

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0
        f[:,0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f[:,1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
        f_arr_norm = (f - self.ideal) / (self.nadir - self.ideal)
        f_arr_norm[:, 0] = 0.5 * f_arr_norm[:, 0]
        return f_arr_norm


    def _evaluate_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0
        f1 = L * ( (2 * x1) + np.sqrt(2.0) * x2 + torch.sqrt(x3) + x4 )
        f2 = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))
        f_arr = torch.stack((f1, f2), dim=1)
        f_arr_norm = (f_arr - Tensor(self.ideal).to(f_arr.device) ) / Tensor(self.nadir - self.ideal).to(f_arr.device)
        f_arr_norm[:, 0] = 0.5 * f_arr_norm[:, 0]
        return f_arr_norm


class RE22(BaseMOP):
    def __init__(self, n_var=3, n_obj=2, lbound=np.zeros(30),
                 ubound=np.ones(30)):
        self.n_var=n_var
        self.n_obj=n_obj
        self.problem_name = 'RE22'
        self.n_cons = 0
        self.n_original_constraints = 2

        self.ideal = np.array([5.88, 0.0])
        self.nadir = np.array([361.262944647, 180.01547])


        self.ubound = np.zeros(self.n_var)
        self.lbound = np.zeros(self.n_var)

        self.lbound[0] = 0.2
        self.lbound[1] = 0.0
        self.lbound[2] = 0.0
        self.ubound[0] = 15
        self.ubound[1] = 20
        self.ubound[2] = 40

        self.n_var = n_var
        self.n_obj = n_obj

        self.feasible_vals = np.array(
            [0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60,
             1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3, 10, 3.16, 3.41,
             3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53,
             5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85,
             12.0, 13.0, 14.0, 15.0])


    def _evaluate_numpy(self, x):
        n_sub = len(x)
        f = np.zeros( (n_sub, self.n_obj) )
        g = np.zeros( (n_sub, self.n_original_constraints)  )
        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx_arr = [np.abs(np.asarray(self.feasible_vals) - x0).argmin() for x0 in x[:,0]]
        x1 = array([self.feasible_vals[idx] for idx in idx_arr])
        x2 = x[:,1]
        x3 = x[:,2]
        # First original objective function
        f[:,0] = (29.4 * x1) + (0.6 * x2 * x3)
        # Original constraint functions
        g[:,0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
        g[:,1] = 4.0 - (x3 / x2)
        g = np.where(g < 0, -g, 0)
        f[:,1] = g[:,0] + g[:,1]
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        f_norm[:, 0] = 0.5 * f_norm[:, 0]
        return f_norm

    def _evaluate_torch(self, x):
        '''
            Maybe some bugs. Convergence is not good.
        '''
        n_sub = len(x)
        f = torch.zeros((n_sub, self.n_obj))
        g = torch.zeros((n_sub, self.n_original_constraints))

        # Finding the nearest feasible value
        feasible_vals_tensor = torch.tensor(self.feasible_vals)
        x0 = x[:, 0].unsqueeze(1)
        idx_arr = torch.abs(feasible_vals_tensor - x0).argmin(dim=1)
        x1 = feasible_vals_tensor[idx_arr]
        x2 = x[:, 1]
        x3 = x[:, 2]
        # First original objective function
        f[:, 0] = (29.4 * x1) + (0.6 * x2 * x3)

        # Original constraint functions
        g[:, 0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
        g[:, 1] = 4.0 - (x3 / x2)

        # Apply constraint penalties
        g = torch.where(g < 0, -g, torch.zeros_like(g))

        # Second objective function
        f[:, 1] = g[:, 0] + g[:, 1]

        # Normalization
        ideal_tensor = torch.tensor(self.ideal)
        nadir_tensor = torch.tensor(self.nadir)
        f_norm = (f - ideal_tensor) / (nadir_tensor - ideal_tensor)
        f_norm[:, 0] = 0.5 * f_norm[:, 0]

        return f_norm


class RE23(BaseMOP):
    def __init__(self, n_var=4, n_obj=2, lbound=np.zeros(2),
                 ubound=np.ones(2)):
        self.problem_name = 'RE23'
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_cons = 0
        self.n_original_constraints = 3

        self.ideal = array([15.9018007813, 0.0])
        self.nadir = array([481.608088535, 44.2819047619])

        self.ubound = np.zeros(self.n_var)
        self.lbound = np.zeros(self.n_var)
        self.lbound[0] = 1
        self.lbound[1] = 1
        self.lbound[2] = 10
        self.lbound[3] = 10
        self.ubound[0] = 100
        self.ubound[1] = 100
        self.ubound[2] = 200
        self.ubound[3] = 240

    def _evaluate_numpy(self, x):
        f = np.zeros( (len(x), self.n_obj) )
        g = np.zeros( (len(x), self.n_original_constraints))
        x1 = 0.0625 * np.round(x[:,0]).astype(np.int32)
        x2 = 0.0625 * np.round(x[:,1]).astype(np.int32)
        x3 = x[:,2]
        x4 = x[:,3]
        # First original objective function
        f[:,0] = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        # Original constraint functions
        g[:,0] = x1 - (0.0193 * x3)
        g[:,1] = x2 - (0.00954 * x3)
        g[:,2] = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        g = np.where(g < 0, -g, 0)
        f[:,1] = g[:,0] + g[:,1] + g[:,2]
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f_norm


class RE24(BaseMOP):
    def __init__(self, n_var=2, n_obj=2, lbound=np.zeros(2),
                 ubound=np.ones(2)):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         lbound=lbound,
                         ubound=ubound, )
        self.problem_name = 'RE24'
        self.n_obj = 2
        self.n_var = 2
        self.n_cons = 0
        self.n_original_constraints = 4
        self.ubound = np.zeros(self.n_var)
        self.lbound = np.zeros(self.n_var)
        self.lbound[0] = 0.5
        self.lbound[1] = 0.5
        self.ubound[0] = 4
        self.ubound[1] = 50
        self.ideal = np.array([60.5, 0.0])
        self.nadir = np.array([481.608088535, 44.2819047619])


    def _evaluate_numpy(self, x):
        n_sub = len(x)
        # f = np.zeros(self.n_objectives)
        g = np.zeros( (n_sub, self.n_original_constraints) )
        x1 = x[:,0]
        x2 = x[:,1]
        # First original objective function
        f1 = x1 + (120 * x2)
        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)
        g[:,0] = 1 - (sigma_b / sigma_b_max)
        g[:,1] = 1 - (tau / tau_max)
        g[:,2] = 1 - (delta / delta_max)
        g[:,3] = 1 - (sigma_b / sigma_k)
        g = np.where(g < 0, -g, 0)
        f2 = g[:,0] + g[:,1] + g[:,2] + g[:,3]
        f_arr = np.stack((f1, f2), axis=1)
        f_norm = (f_arr - self.ideal) / (self.nadir - self.ideal)
        return f_norm


    def _evaluate_torch(self, x):
        pass


class RE25(BaseMOP):
    def __init__(self, n_var=3, n_obj=2):
        self.problem_name = 'RE25'
        self.n_obj = n_obj
        self.n_var = n_var

        self.n_cons = 0
        self.n_original_constraints = 6

        self.ideal = array([0.037591349242869145, 0.0])
        self.nadir = array([0.40397042546, 2224669.22419])

        self.ubound = np.zeros( self.n_var )
        self.lbound = np.zeros( self.n_var )
        self.lbound[0] = 1
        self.lbound[1] = 0.6
        self.lbound[2] = 0.09
        self.ubound[0] = 70
        self.ubound[1] = 3
        self.ubound[2] = 0.5

        self.feasible_vals = np.array(
            [0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025,
             0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162,
             0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])

    def _evaluate_numpy(self, x):
        n_sub = len(x)
        f = np.zeros( (n_sub, self.n_obj) )
        g = np.zeros( (n_sub, self.n_original_constraints) )
        x1 = np.round(x[:,0])
        x2 = x[:,1]

        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx_array = array([np.abs(np.asarray(self.feasible_vals) - x2).argmin() for x2 in x[:,2]])

        x3 = array( [self.feasible_vals[idx] for idx in idx_array] )

        # first original objective function
        f[:,0] = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0

        # constraint functions
        Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
        Fmax = 1000.0
        S = 189000.0
        G = 11.5 * 1e+6
        K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
        lmax = 14.0
        lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
        dmin = 0.2
        Dmax = 3
        Fp = 300.0
        sigmaP = Fp / K
        sigmaPM = 6
        sigmaW = 1.25

        g[:,0] = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
        g[:,1] = -lf + lmax
        g[:,2] = -3 + (x2 / x3)
        g[:,3] = -sigmaP + sigmaPM
        g[:,4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
        g[:,5] = sigmaW - ((Fmax - Fp) / K)

        g = np.where(g < 0, -g, 0)
        f[:,1] = g[:,0] + g[:,1] + g[:,2] + g[:,3] + g[:,4] + g[:,5]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f_norm

    def _evaluate_torch(self, x):
        pass


class RE31(BaseMOP):
    def __init__(self, n_obj=3, n_var=3):
        self.problem_name = 'RE31'
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_cons = 0
        self.n_original_constraints = 3
        self.ubound = np.zeros(self.n_var)
        self.lbound = np.zeros(self.n_var)
        self.lbound[0] = 0.00001
        self.lbound[1] = 0.00001
        self.lbound[2] = 1.0
        self.ubound[0] = 100.0
        self.ubound[1] = 100.0
        self.ubound[2] = 3.0
        self.ideal = np.array([5.53731918799e-05, 0.333333333333, 0.0])
        self.nadir = np.array([500.002668442, 8246211.25124, 19359919.7502])

    def _evaluate_numpy(self, x):
        n_sub = len(x)
        f = np.zeros( (n_sub, self.n_obj) )
        g = np.zeros( (n_sub, self.n_original_constraints) )
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        # First original objective function
        f[:,0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f[:,1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)
        # Constraint functions
        g[:,0] = 0.1 - f[:,0]
        g[:,1] = 100000.0 - f[:,1]
        g[:,2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g = np.where(g < 0, -g, 0)
        f[:,2] = g[:,0] + g[:,1] + g[:,2]
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f_norm

    def _evaluate_torch(self, x):
        assert False, '_evaluate_torch not implemented'


class RE37(BaseMOP):
    def __init__(self, n_obj=3, n_var=4):
        self.problem_name = 'RE37'
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_cons = 0
        self.n_original_constraints = 0

        self.lbound = np.full(self.n_var, 0)
        self.ubound = np.full(self.n_var, 1)

        self.ideal = np.array([0.00889341391106, 0.00488, -0.431499999825])
        self.nadir = np.array([0.98949120096, 0.956587924661, 0.987530948586])

    def _evaluate_numpy(self, x):
        n_sub = len(x)
        f = np.zeros( (n_sub, self.n_obj) )

        xAlpha = x[:,0]
        xHA = x[:,1]
        xOA = x[:,2]
        xOPTT = x[:,3]

        # f1 (TF_max)
        f[:,0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (
                    0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (
                           0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (
                           0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (
                           0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f[:,1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (
                    0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (
                           0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (
                           0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (
                           0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f[:,2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (
                    0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (
                           0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (
                           0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (
                           0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                           0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                           0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f_norm

    def _evaluate_torch(self, x):
        n_sub = x.shape[0]
        f = torch.zeros((n_sub, self.n_obj)).to(x.device)

        xAlpha = x[:, 0]
        xHA = x[:, 1]
        xOA = x[:, 2]
        xOPTT = x[:, 3]

        # f1 (TF_max)
        f[:, 0] = (
                0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT)
                - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA)
                - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA)
                - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA)
                + (0.0184 * xOPTT * xOPTT)
        )

        # f2 (X_cc)
        f[:, 1] = (
                0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT)
                + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA)
                - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA)
                + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA)
                + (0.0192 * xOPTT * xOPTT)
        )

        # f3 (TT_max)
        f[:, 2] = (
                0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT)
                - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA)
                + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA)
                + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT)
                + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha)
                - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA)
                + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha)
                - (0.281 * xHA * xAlpha * xOA)
        )

        f_norm = (f - Tensor(self.ideal).to(f.device) ) / (Tensor(self.nadir - self.ideal).to(f.device))
        return f_norm




class RE41(BaseMOP):
    def __init__(self, n_obj=4, n_var=7):
        self.problem_name = 'RE41'
        self.n_obj = n_obj
        self.n_var = n_var
        self.n_cons = 0
        self.n_original_constraints = 10

        self.lbound = np.zeros(self.n_var)
        self.ubound = np.zeros(self.n_var)
        self.lbound[0] = 0.5
        self.lbound[1] = 0.45
        self.lbound[2] = 0.5
        self.lbound[3] = 0.5
        self.lbound[4] = 0.875
        self.lbound[5] = 0.4
        self.lbound[6] = 0.4
        self.ubound[0] = 1.5
        self.ubound[1] = 1.35
        self.ubound[2] = 1.5
        self.ubound[3] = 1.5
        self.ubound[4] = 2.625
        self.ubound[5] = 1.2
        self.ubound[6] = 1.2

        self.ideal = np.array([15.576004, 3.58525, 10.61064375, 0.0])
        self.nadir = np.array([39.2905121788, 4.42725, 13.09138125, 9.49401929991])


    def _evaluate_numpy(self, x):
        n_sub = len(x)

        f = np.zeros( (n_sub, self.n_obj) )
        g = np.zeros( (n_sub, self.n_original_constraints) )

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]

        # First original objective function
        f[:,0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second original objective function
        f[:,1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        # Third original objective function
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[:,2] = 0.5 * (Vmbp + Vfd)

        # Constraint functions
        g[:,0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[:,1] = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g[:,2] = 0.32 - (
                    0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g[:,3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[:,4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[:,5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[:,6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[:,7] = 4 - f[:,1]
        g[:,8] = 9.9 - Vmbp
        g[:,9] = 15.7 - Vfd
        g = np.where(g < 0, -g, 0)
        f[:,3] = g[:,0] + g[:,1] + g[:,2] + g[:,3] + g[:,4] + g[:,5] + g[:,6] + g[:,7] + g[:,8] + g[:,9]
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f_norm

    def _evaluate_torch(self, x):
        pass


class RE42(BaseMOP):
    def __init__(self):
        self.problem_name = 'RE42'
        self.n_obj = 4
        self.n_var = 6
        self.n_cons = 0
        self.n_original_constraints = 9
        self.lbound = np.zeros(self.n_var )
        self.ubound = np.zeros(self.n_var )
        self.lbound[0] = 150.0
        self.lbound[1] = 20.0
        self.lbound[2] = 13.0
        self.lbound[3] = 10.0
        self.lbound[4] = 14.0
        self.lbound[5] = 0.63
        self.ubound[0] = 274.32
        self.ubound[1] = 32.31
        self.ubound[2] = 25.0
        self.ubound[3] = 11.71
        self.ubound[4] = 18.0
        self.ubound[5] = 0.75
        self.ideal = np.array([-2756.2590400638524, 3962.557843228888, 1947.880856925791, 0.0])
        self.nadir = np.array([-1010.5229595219643, 13827.138456300128, 2611.9668107424536, 12.437669929732023 ])


    def _evaluate_numpy(self, x):
        n_sub = len(x)

        f = np.zeros( (n_sub, self.n_obj) )
        # NOT g
        constraintFuncs = np.zeros( (n_sub, self.n_original_constraints) )

        x_L = x[:,0]
        x_B = x[:,1]
        x_D = x[:,2]
        x_T = x[:,3]
        x_Vk = x[:,4]
        x_CB = x[:,5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (
                    2400.0 * np.power(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * np.power(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f[:,0] = annual_costs / annual_cargo
        f[:,1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[:,2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[:,0] = (x_L / x_B) - 6.0
        constraintFuncs[:,1] = -(x_L / x_D) + 15.0
        constraintFuncs[:,2] = -(x_L / x_T) + 19.0
        constraintFuncs[:,3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[:,4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[:,5] = 500000.0 - DWT
        constraintFuncs[:,6] = DWT - 3000.0
        constraintFuncs[:,7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[:,8] = (KB + BMT - KG) - (0.07 * x_B)

        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)
        f[:,3] = constraintFuncs[:,0] + constraintFuncs[:,1] + constraintFuncs[:,2] + constraintFuncs[:,3] + constraintFuncs[:,4] + \
               constraintFuncs[:,5] + constraintFuncs[:,6] + constraintFuncs[:,7] + constraintFuncs[:,8]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        return f





