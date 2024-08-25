#!/usr/bin/env python
"""
  A real-world multimnist-objective problem suite (the RE benchmark set)
  Reference:
  Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective Problem Suite" Applied Soft Computing. 89: 106078 (2020)
   Copyright (c) 2020 Ryoji Tanabe

  I re-implemented the RE problem set by referring to its C source code (reproblem.c). While variables directly copied from the C source code are written in CamelCase, the other variables are written in snake_case. It is somewhat awkward.

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

class RE21():
    def __init__(self, n_var=4, n_obj=2, lower_bound=np.zeros(30),
                 upper_bound=np.ones(30)):
        self.problem_name = 'RE21'
        self.n_constraints = 0
        self.n_original_constraints = 0

        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        self.ubound = np.full(self.n_variables, 3 * tmp_val)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = tmp_val
        self.lbound[1] = np.sqrt(2.0) * tmp_val
        self.lbound[2] = np.sqrt(2.0) * tmp_val
        self.lbound[3] = tmp_val

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        F = 10.0
        sigma = 10.0
        E = 2.0 * 1e5
        L = 200.0

        f[0] = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f[1] = ((F * L) / E) * ((2.0 / x1) + (2.0 * np.sqrt(2.0) / x2) - (2.0 * np.sqrt(2.0) / x3) + (2.0 / x4))

        return f


class RE22():
    def __init__(self):
        self.problem_name = 'RE22'
        self.n_objectives = 2
        self.n_variables = 3

        self.n_constraints = 0
        self.n_original_constraints = 2

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 0.2
        self.lbound[1] = 0.0
        self.lbound[2] = 0.0
        self.ubound[0] = 15
        self.ubound[1] = 20
        self.ubound[2] = 40

        self.feasible_vals = np.array(
            [0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60,
             1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3, 10, 3.16, 3.41,
             3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53,
             5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85,
             12.0, 13.0, 14.0, 15.0])

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)
        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx = np.abs(np.asarray(self.feasible_vals) - x[0]).argmin()
        x1 = self.feasible_vals[idx]
        x2 = x[1]
        x3 = x[2]

        # First original objective function
        f[0] = (29.4 * x1) + (0.6 * x2 * x3)

        # Original constraint functions
        g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
        g[1] = 4.0 - (x3 / x2)
        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1]

        return f


class RE23():
    def __init__(self):
        self.problem_name = 'RE23'
        self.n_objectives = 2
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 3

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 1
        self.lbound[1] = 1
        self.lbound[2] = 10
        self.lbound[3] = 10
        self.ubound[0] = 100
        self.ubound[1] = 100
        self.ubound[2] = 200
        self.ubound[3] = 240

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = 0.0625 * int(np.round(x[0]))
        x2 = 0.0625 * int(np.round(x[1]))
        x3 = x[2]
        x4 = x[3]

        # First original objective function
        f[0] = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)

        # Original constraint functions
        g[0] = x1 - (0.0193 * x3)
        g[1] = x2 - (0.00954 * x3)
        g[2] = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1] + g[2]

        return f


class RE24():
    def __init__(self):
        self.problem_name = 'RE24'
        self.n_objectives = 2
        self.n_variables = 2
        self.n_constraints = 0
        self.n_original_constraints = 4

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 0.5
        self.lbound[1] = 0.5
        self.ubound[0] = 4
        self.ubound[1] = 50

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]

        # First original objective function
        f[0] = x1 + (120 * x2)

        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)

        g[0] = 1 - (sigma_b / sigma_b_max)
        g[1] = 1 - (tau / tau_max)
        g[2] = 1 - (delta / delta_max)
        g[3] = 1 - (sigma_b / sigma_k)
        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1] + g[2] + g[3]

        return f


class RE25():
    def __init__(self):
        self.problem_name = 'RE25'
        self.n_objectives = 2
        self.n_variables = 3
        self.n_constraints = 0
        self.n_original_constraints = 6

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
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

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = np.round(x[0])
        x2 = x[1]
        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx = np.abs(np.asarray(self.feasible_vals) - x[2]).argmin()
        x3 = self.feasible_vals[idx]

        # first original objective function
        f[0] = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0

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

        g[0] = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
        g[1] = -lf + lmax
        g[2] = -3 + (x2 / x3)
        g[3] = -sigmaP + sigmaPM
        g[4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
        g[5] = sigmaW - ((Fmax - Fp) / K)

        g = np.where(g < 0, -g, 0)
        f[1] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5]

        return f


class RE31():
    def __init__(self):
        self.problem_name = 'RE31'
        self.n_objectives = 3
        self.n_variables = 3
        self.n_constraints = 0
        self.n_original_constraints = 3

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 0.00001
        self.lbound[1] = 0.00001
        self.lbound[2] = 1.0
        self.ubound[0] = 100.0
        self.ubound[1] = 100.0
        self.ubound[2] = 3.0

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        # First original objective function
        f[0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f[1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Constraint functions
        g[0] = 0.1 - f[0]
        g[1] = 100000.0 - f[1]
        g[2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g = np.where(g < 0, -g, 0)
        f[2] = g[0] + g[1] + g[2]

        return f


class RE32():
    def __init__(self):
        self.problem_name = 'RE32'
        self.n_objectives = 3
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 4

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 0.125
        self.lbound[1] = 0.1
        self.lbound[2] = 0.1
        self.lbound[3] = 0.125
        self.ubound[0] = 5.0
        self.ubound[1] = 10.0
        self.ubound[2] = 10.0
        self.ubound[3] = 5.0

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # First original objective function
        f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = np.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g[0] = tauMax - tau
        g[1] = sigmaMax - sigma
        g[2] = x4 - x1
        g[3] = PC - P
        g = np.where(g < 0, -g, 0)
        f[2] = g[0] + g[1] + g[2] + g[3]

        return f


class RE33():
    def __init__(self):
        self.problem_name = 'RE33'
        self.n_objectives = 3
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 4

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 55
        self.lbound[1] = 75
        self.lbound[2] = 1000
        self.lbound[3] = 11
        self.ubound[0] = 80
        self.ubound[1] = 110
        self.ubound[2] = 3000
        self.ubound[3] = 20

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        # First original objective function
        f[0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f[1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g[0] = (x2 - x1) - 20.0
        g[1] = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g[2] = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
        g[3] = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
        g = np.where(g < 0, -g, 0)
        f[2] = g[0] + g[1] + g[2] + g[3]

        return f


class RE34():
    def __init__(self):
        self.problem_name = 'RE34'
        self.n_objectives = 3
        self.n_variables = 5
        self.n_constraints = 0
        self.n_original_constraints = 0

        self.lbound = np.full(self.n_variables, 1)
        self.ubound = np.full(self.n_variables, 3)

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]

        f[0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f[1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (
                    0.0861 * x1 * x5) + (0.3628 * x2 * x4) - (0.1106 * x1 * x1) - (0.3437 * x3 * x3) + (
                           0.1764 * x4 * x4)
        f[2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (
                    0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)

        return f


class RE35():
    def __init__(self):
        self.problem_name = 'RE35'
        self.n_objectives = 3
        self.n_variables = 7
        self.n_constraints = 0
        self.n_original_constraints = 11

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
        self.lbound[0] = 2.6
        self.lbound[1] = 0.7
        self.lbound[2] = 17
        self.lbound[3] = 7.3
        self.lbound[4] = 7.3
        self.lbound[5] = 2.9
        self.lbound[6] = 5.0
        self.ubound[0] = 3.6
        self.ubound[1] = 0.8
        self.ubound[2] = 28
        self.ubound[3] = 8.3
        self.ubound[4] = 8.3
        self.ubound[5] = 3.9
        self.ubound[6] = 5.5

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = np.round(x[2])
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        # First original objective function (weight)
        f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (
                    x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

        # Second original objective function (stress)
        tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
        f[1] = np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        # Constraint functions
        g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g[4] = -(x2 * x3) + 40.0
        g[5] = -(x1 / x2) + 12.0
        g[6] = -5.0 + (x1 / x2)
        g[7] = -1.9 + x4 - 1.5 * x6
        g[8] = -1.9 + x5 - 1.1 * x7
        g[9] = -f[1] + 1300.0
        tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
        g = np.where(g < 0, -g, 0)
        f[2] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9] + g[10]

        return f


class RE36():
    def __init__(self):
        self.problem_name = 'RE36'
        self.n_objectives = 3
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 1

        self.lbound = np.full(self.n_variables, 12)
        self.ubound = np.full(self.n_variables, 60)

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        # all the four variables must be inverger values
        x1 = np.round(x[0])
        x2 = np.round(x[1])
        x3 = np.round(x[2])
        x4 = np.round(x[3])

        # First original objective function
        f[0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = [x1, x2, x3, x4]
        f[1] = max(l)

        g[0] = 0.5 - (f[0] / 6.931)
        g = np.where(g < 0, -g, 0)
        f[2] = g[0]

        return f


class RE37():
    def __init__(self):
        self.problem_name = 'RE37'
        self.n_objectives = 3
        self.n_variables = 4
        self.n_constraints = 0
        self.n_original_constraints = 0

        self.lbound = np.full(self.n_variables, 0)
        self.ubound = np.full(self.n_variables, 1)

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)

        xAlpha = x[0]
        xHA = x[1]
        xOA = x[2]
        xOPTT = x[3]

        # f1 (TF_max)
        f[0] = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (
                    0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (
                           0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (
                           0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (
                           0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f[1] = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (
                    0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (
                           0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (
                           0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (
                           0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f[2] = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (
                    0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (
                           0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (
                           0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (
                           0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                           0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                           0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        return f


class RE41():
    def __init__(self):
        self.problem_name = 'RE41'
        self.n_objectives = 4
        self.n_variables = 7
        self.n_constraints = 0
        self.n_original_constraints = 10

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
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

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        # First original objective function
        f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second original objective function
        f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        # Third original objective function
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[2] = 0.5 * (Vmbp + Vfd)

        # Constraint functions
        g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[1] = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g[2] = 0.32 - (
                    0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g[3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[7] = 4 - f[1]
        g[8] = 9.9 - Vmbp
        g[9] = 15.7 - Vfd

        g = np.where(g < 0, -g, 0)
        f[3] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9]

        return f


class RE42():
    def __init__(self):
        self.problem_name = 'RE42'
        self.n_objectives = 4
        self.n_variables = 6
        self.n_constraints = 0
        self.n_original_constraints = 9

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
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

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        # NOT g
        constraintFuncs = np.zeros(self.n_original_constraints)

        x_L = x[0]
        x_B = x[1]
        x_D = x[2]
        x_T = x[3]
        x_Vk = x[4]
        x_CB = x[5]

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

        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[0] = (x_L / x_B) - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = (KB + BMT - KG) - (0.07 * x_B)

        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)
        f[3] = constraintFuncs[0] + constraintFuncs[1] + constraintFuncs[2] + constraintFuncs[3] + constraintFuncs[4] + \
               constraintFuncs[5] + constraintFuncs[6] + constraintFuncs[7] + constraintFuncs[8]

        return f


class RE61():
    def __init__(self):
        self.problem_name = 'RE61'
        self.n_objectives = 6
        self.n_variables = 3
        self.n_constraints = 0
        self.n_original_constraints = 7

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
        self.lbound[0] = 0.01
        self.lbound[1] = 0.01
        self.lbound[2] = 0.01
        self.ubound[0] = 0.45
        self.ubound[1] = 0.10
        self.ubound[2] = 0.10

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        # First original objective function
        f[0] = 106780.37 * (x[1] + x[2]) + 61704.67
        # Second original objective function
        f[1] = 3000 * x[0]
        # Third original objective function
        f[2] = 305700 * 2289 * x[1] / np.power(0.06 * 2289, 0.65)
        # Fourth original objective function
        f[3] = 250 * 2289 * np.exp(-39.75 * x[1] + 9.9 * x[2] + 2.74)
        # Fifth original objective function
        f[4] = 25 * (1.39 / (x[0] * x[1]) + 4940 * x[2] - 80)

        # Constraint functions
        g[0] = 1 - (0.00139 / (x[0] * x[1]) + 4.94 * x[2] - 0.08)
        g[1] = 1 - (0.000306 / (x[0] * x[1]) + 1.082 * x[2] - 0.0986)
        g[2] = 50000 - (12.307 / (x[0] * x[1]) + 49408.24 * x[2] + 4051.02)
        g[3] = 16000 - (2.098 / (x[0] * x[1]) + 8046.33 * x[2] - 696.71)
        g[4] = 10000 - (2.138 / (x[0] * x[1]) + 7883.39 * x[2] - 705.04)
        g[5] = 2000 - (0.417 * x[0] * x[1] + 1721.26 * x[2] - 136.54)
        g[6] = 550 - (0.164 / (x[0] * x[1]) + 631.13 * x[2] - 54.48)

        g = np.where(g < 0, -g, 0)
        f[5] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6]

        return f


class RE91():
    def __init__(self):
        self.problem_name = 'RE91'
        self.n_objectives = 9
        self.n_variables = 7
        self.n_constraints = 0
        self.n_original_constraints = 0

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
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

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        # stochastic variables
        x8 = 0.006 * (np.random.normal(0, 1)) + 0.345
        x9 = 0.006 * (np.random.normal(0, 1)) + 0.192
        x10 = 10 * (np.random.normal(0, 1)) + 0.0
        x11 = 10 * (np.random.normal(0, 1)) + 0.0

        # First function
        f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.75 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second function
        f[1] = max(0.0, (1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) / 1.0)
        # Third function
        f[2] = max(0.0, (
                    0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.87570001 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11) / 0.32)
        # Fourth function
        f[3] = max(0.0, (
                    0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 * x2) / 0.32)
        # Fifth function
        f[4] = max(0.0, (
                    0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2) / 0.32)
        # Sixth function
        tmp = ((
                           28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8 + 0.32 * x9 * x10) + (
                           33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) + (
                           46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10)) / 3
        f[5] = max(0.0, tmp / 32)
        # Seventh function
        f[6] = max(0.0, (
                    4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11) / 4.0)
        # EighthEighth function
        f[7] = max(0.0, (
                    10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 + 0.02054 * x3 * x10 - 0.0198 * x4 * x10 + 0.028 * x6 * x10) / 9.9)
        # Ninth function
        f[8] = max(0.0, (
                    16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 * x11) / 15.7)

        return f


class CRE21():
    def __init__(self):
        self.problem_name = 'CRE21'
        self.n_objectives = 2
        self.n_variables = 3
        self.n_constraints = 3

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 0.00001
        self.lbound[1] = 0.00001
        self.lbound[2] = 1.0
        self.ubound[0] = 100.0
        self.ubound[1] = 100.0
        self.ubound[2] = 3.0

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        # First original objective function
        f[0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f[1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Constraint functions
        g[0] = 0.1 - f[0]
        g[1] = 100000.0 - f[1]
        g[2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g = np.where(g < 0, -g, 0)

        return f, g


class CRE22():
    def __init__(self):
        self.problem_name = 'CRE22'
        self.n_objectives = 2
        self.n_variables = 4
        self.n_constraints = 4

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 0.125
        self.lbound[1] = 0.1
        self.lbound[2] = 0.1
        self.lbound[3] = 0.125
        self.ubound[0] = 5.0
        self.ubound[1] = 10.0
        self.ubound[2] = 10.0
        self.ubound[3] = 5.0

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # First original objective function
        f[0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f[1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = np.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g[0] = tauMax - tau
        g[1] = sigmaMax - sigma
        g[2] = x4 - x1
        g[3] = PC - P
        g = np.where(g < 0, -g, 0)

        return f, g


class CRE23():
    def __init__(self):
        self.problem_name = 'CRE23'
        self.n_objectives = 2
        self.n_variables = 4
        self.n_constraints = 4

        self.ubound = np.zeros(self.n_variables)
        self.lbound = np.zeros(self.n_variables)
        self.lbound[0] = 55
        self.lbound[1] = 75
        self.lbound[2] = 1000
        self.lbound[3] = 11
        self.ubound[0] = 80
        self.ubound[1] = 110
        self.ubound[2] = 3000
        self.ubound[3] = 20

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        # First original objective function
        f[0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f[1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g[0] = (x2 - x1) - 20.0
        g[1] = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g[2] = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
        g[3] = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
        g = np.where(g < 0, -g, 0)

        return f, g


class CRE24():
    def __init__(self):
        self.problem_name = 'CRE24'
        self.n_objectives = 2
        self.n_variables = 7
        self.n_constraints = 11

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)

        self.lbound[0] = 2.6
        self.lbound[1] = 0.7
        self.lbound[2] = 17
        self.lbound[3] = 7.3
        self.lbound[4] = 7.3
        self.lbound[5] = 2.9
        self.lbound[6] = 5.0
        self.ubound[0] = 3.6
        self.ubound[1] = 0.8
        self.ubound[2] = 28
        self.ubound[3] = 8.3
        self.ubound[4] = 8.3
        self.ubound[5] = 3.9
        self.ubound[6] = 5.5

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = np.round(x[2])
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        # First original objective function (weight)
        f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (
                    x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

        # Second original objective function (stress)
        tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
        f[1] = np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        # Constraint functions
        g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g[4] = -(x2 * x3) + 40.0
        g[5] = -(x1 / x2) + 12.0
        g[6] = -5.0 + (x1 / x2)
        g[7] = -1.9 + x4 - 1.5 * x6
        g[8] = -1.9 + x5 - 1.1 * x7
        g[9] = -f[1] + 1300.0
        tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
        g = np.where(g < 0, -g, 0)

        return f, g


class CRE25():
    def __init__(self):
        self.problem_name = 'CRE25'
        self.n_objectives = 2
        self.n_variables = 4
        self.n_constraints = 1

        self.lbound = np.full(self.n_variables, 12)
        self.ubound = np.full(self.n_variables, 60)

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_constraints)

        # all the four variables must be inverger values
        x1 = np.round(x[0])
        x2 = np.round(x[1])
        x3 = np.round(x[2])
        x4 = np.round(x[3])

        # First original objective function
        f[0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = [x1, x2, x3, x4]
        f[1] = max(l)

        g[0] = 0.5 - (f[0] / 6.931)
        g = np.where(g < 0, -g, 0)

        return f, g


class CRE31():
    def __init__(self):
        self.problem_name = 'CRE31'
        self.n_objectives = 3
        self.n_variables = 7
        self.n_constraints = 10

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
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

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        # First original objective function
        f[0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second original objective function
        f[1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        # Third original objective function
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[2] = 0.5 * (Vmbp + Vfd)

        # Constraint functions
        g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[1] = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g[2] = 0.32 - (
                    0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g[3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[7] = 4 - f[1]
        g[8] = 9.9 - Vmbp
        g[9] = 15.7 - Vfd
        g = np.where(g < 0, -g, 0)

        return f, g


class CRE32():
    def __init__(self):
        self.problem_name = 'CRE32'
        self.n_objectives = 3
        self.n_variables = 6
        self.n_constraints = 9

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
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

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        # NOT g
        constraintFuncs = np.zeros(self.n_constraints)

        x_L = x[0]
        x_B = x[1]
        x_D = x[2]
        x_T = x[3]
        x_Vk = x[4]
        x_CB = x[5]

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

        f[0] = annual_costs / annual_cargo
        f[1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[0] = (x_L / x_B) - 6.0
        constraintFuncs[1] = -(x_L / x_D) + 15.0
        constraintFuncs[2] = -(x_L / x_T) + 19.0
        constraintFuncs[3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[5] = 500000.0 - DWT
        constraintFuncs[6] = DWT - 3000.0
        constraintFuncs[7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[8] = (KB + BMT - KG) - (0.07 * x_B)
        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)

        return f, constraintFuncs


class CRE51():
    def __init__(self):
        self.problem_name = 'CRE51'
        self.n_objectives = 5
        self.n_variables = 3
        self.n_constraints = 7

        self.lbound = np.zeros(self.n_variables)
        self.ubound = np.zeros(self.n_variables)
        self.lbound[0] = 0.01
        self.lbound[1] = 0.01
        self.lbound[2] = 0.01
        self.ubound[0] = 0.45
        self.ubound[1] = 0.10
        self.ubound[2] = 0.10

    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_constraints)

        # First original objective function
        f[0] = 106780.37 * (x[1] + x[2]) + 61704.67
        # Second original objective function
        f[1] = 3000 * x[0]
        # Third original objective function
        f[2] = 305700 * 2289 * x[1] / np.power(0.06 * 2289, 0.65)
        # Fourth original objective function
        f[3] = 250 * 2289 * np.exp(-39.75 * x[1] + 9.9 * x[2] + 2.74)
        # Fifth original objective function
        f[4] = 25 * (1.39 / (x[0] * x[1]) + 4940 * x[2] - 80)

        # Constraint functions
        g[0] = 1 - (0.00139 / (x[0] * x[1]) + 4.94 * x[2] - 0.08)
        g[1] = 1 - (0.000306 / (x[0] * x[1]) + 1.082 * x[2] - 0.0986)
        g[2] = 50000 - (12.307 / (x[0] * x[1]) + 49408.24 * x[2] + 4051.02)
        g[3] = 16000 - (2.098 / (x[0] * x[1]) + 8046.33 * x[2] - 696.71)
        g[4] = 10000 - (2.138 / (x[0] * x[1]) + 7883.39 * x[2] - 705.04)
        g[5] = 2000 - (0.417 * x[0] * x[1] + 1721.26 * x[2] - 136.54)
        g[6] = 550 - (0.164 / (x[0] * x[1]) + 631.13 * x[2] - 54.48)
        g = np.where(g < 0, -g, 0)

        return f, g




if __name__ == '__main__':
    np.random.seed(seed=1)
    fun = RE21()

    x = fun.lbound + (fun.ubound - fun.lbound) * np.random.rand(fun.n_variables)
    print("Problem = {}".format(fun.problem_name))
    print("Number of objectives = {}".format(fun.n_objectives))
    print("Number of variables = {}".format(fun.n_variables))
    print("Number of constraints = {}".format(fun.n_constraints))
    print("Lower bounds = ", fun.lbound)
    print("Upper bounds = ", fun.ubound)
    print("x = ", x)

    if 'CRE' in fun.problem_name:
        f, g = fun.evaluate(x)
        print("f(x) = {}".format(f))
        print("g(x) = {}".format(g))
    else:
        f = fun.evaluate(x)
        print("f(x) = {}".format(f))