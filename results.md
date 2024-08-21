**Table 1. Mean and standard deviation on VLMOP1 problem**

| Method       | Lmin              | Soft Lmin         | Spacing           | Sparsity          | HV                | IP                | Cross Angle       | PBI               |
|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| EPO          | 0.162 (0.000)     | 0.061 (0.000)     | 0.029 (0.001)     | 0.043 (0.000)     | 0.283 (0.000)     | 0.776 (0.000)     | **0.046 (0.041)** | **0.930 (0.003)** |
| MGDA-UB      | 0.012 (0.013)     | -0.098 (0.011)    | 0.036 (0.011)     | **0.006 (0.001)** | 0.228 (0.008)     | 0.606 (0.010)     | 31.278 (1.533)    | 2.986 (0.088)     |
| PMGDA        | 0.150 (0.001)     | 0.055 (0.000)     | 0.034 (0.001)     | 0.042 (0.000)     | 0.283 (0.000)     | 0.775 (0.000)     | 0.318 (0.037)     | 0.952 (0.003)     |
| Random       | 0.000 (0.000)     | -0.161 (0.002)    | **0.000 (0.000)** | 0.272 (0.000)     | 0.044 (0.000)     | 0.410 (0.127)     | 52.290 (12.938)   | 3.894 (0.590)     |
| MOO-SVGD     | 0.060 (0.002)     | -0.077 (0.004)    | 0.033 (0.018)     | 0.009 (0.003)     | 0.212 (0.003)     | 0.633 (0.024)     | 29.647 (3.305)    | 2.963 (0.197)     |
| PMTL         | 0.014 (0.010)     | -0.068 (0.009)    | 0.061 (0.020)     | 0.018 (0.007)     | 0.260 (0.012)     | 0.706 (0.004)     | 15.036 (1.270)    | 1.993 (0.093)     |
| HVGrad       | **0.182 (0.000)** | **0.067 (0.000)** | 0.016 (0.000)     | 0.041 (0.000)     | **0.286 (0.000)** | 0.578 (0.069)     | 34.090 (8.607)    | 3.062 (0.465)     |
| Agg-LS       | 0.000 (0.000)     | -0.159 (0.001)    | 0.002 (0.001)     | 0.272 (0.001)     | 0.043 (0.002)     | **0.227 (0.008)** | 71.168 (0.958)    | 4.764 (0.047)     |
| Agg-Tche     | 0.158 (0.001)     | 0.061 (0.000)     | 0.031 (0.001)     | 0.043 (0.000)     | 0.283 (0.000)     | 0.348 (0.000)     | 55.174 (0.049)    | 3.889 (0.004)     |
| Agg-PBI      | 0.113 (0.074)     | 0.032 (0.046)     | 0.045 (0.030)     | 0.042 (0.002)     | 0.281 (0.002)     | 0.657 (0.097)     | 11.374 (9.125)    | 1.434 (0.402)     |
| Agg-COSMOS   | 0.141 (0.000)     | 0.045 (0.000)     | 0.035 (0.000)     | 0.039 (0.000)     | 0.285 (0.000)     | 0.771 (0.000)     | 1.085 (0.000)     | 1.011 (0.000)     |
| Agg-SoftTche | 0.004 (0.000)     | -0.074 (0.000)    | 0.154 (0.001)     | 0.074 (0.000)     | 0.244 (0.000)     | 0.276 (0.000)     | 63.106 (0.018)    | 4.253 (0.001)     |


Analysis
Table 1 highlights the following key points:

- HV-grad achieves the highest hypervolume due to its gradient-based method designed specifically for hypervolume maximization.
- EPO and PMGDA, which aim to find "exact" Pareto solutions, show the smallest cross angles between preference vectors.
- MGDA-UB, which finds arbitrary Pareto solutions, exhibits high standard deviation, similar to Agg-PBI, which is influenced by its initial solution due to a non-convex term in its objective function.
- EPO, PMGDA, Agg-LS, Agg-Tche, and Agg-SoftTche display relatively low standard deviations, indicating that their solution distributions are less sensitive to initialization.