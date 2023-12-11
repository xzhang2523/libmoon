### Moon: A Standardized/Flexible Framework for MultiObjective OptimizatioN
<img src="moon.png" alt="Moon" width="200">

'''
    I raise my cup to invite the moon.
    With my shadow we become three from single.
'''
-- Li Bai.

Moon: is a multiobjective optimization framework, from single-objective optimization to multiobjective optimization, towards a better understanding of optimization problems.


Do not release.  

Main contributors Xiaoyuan Zhang, Liao Zhang, Weiduo Liao, Xi Lin, Yifan Chen

This project has four important parts:

(1) A standardlized gradient based framework. 

- **Problem** class. 
For synthetic problems, 
   

  | Problem                                          | Paper                                                              | Project/Code                                         |
  |--------------------------------------------------|------------------|------------------------------------------------------|
  | ZDT                                              | [paper](https://ieeexplore.ieee.org/document/996017)               | [project](https://pymoo.org/problems/multi/zdt.html) |
  | DTLZ                                             | Real world problems.                                               | Y                                                    |
  | MAF                                              | [paper](https://link.springer.com/article/10.1007/s40747-017-0039-7) | Y                                                    |
  | [WFG](https://ieeexplore.ieee.org/document/996017) [code]() | Real world problems.                                               | Y                                                    |
  | [Fi's](https://ieeexplore.ieee.org/document/996017) [code]() | Real world problems.                                               | Y                                                    |
  | RE  | [paper](https://arxiv.org/abs/2009.12867)| [code](https://github.com/ryojitanabe/reproblems)|

(2) For multitask learning problems, 

| Problem            | Paper | Project/Code |
|--------------------|------|--------------|
| MO-MNIST           | [PMTL](https://proceedings.neurips.cc/paper_files/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf)     | [COSMOS](https://github.com/ruchtem/cosmos)     |
| Fairness           |[COSMOS](https://arxiv.org/pdf/2103.13392.pdf) |[COSMOS](https://github.com/ruchtem/cosmos) |
| Federated Learning | | |

For 


- **Gradient-based Solver**.

    | Method                                                                                                                                                                              | Property                                                              | #Obj               | Support | Published | Complexity      |
    |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------|---------|----------|-----------------|
    | [EPO](https://proceedings.mlr.press/v119/mahapatra20a/mahapatra20a.pdf) [code](https://github.com/dbmptr/EPOSearch)                                                                 | Exact solution.                                                       | Any                | Y       | ICML 2020 | $O(m^2 n K )$   |
    | [COSMOS](https://arxiv.org/pdf/2103.13392.pdf) [code](https://github.com/ruchtem/cosmos)                                                                                            | Approximated exact solution.                                          | Any                | Y    | ICDM 2021| $O(m n K )$     |
    | [MOO-SVGD](https://openreview.net/pdf?id=S2-j0ZegyrE) [code](https://github.com/gnobitab/MultiObjectiveSampling)                                                                    | A set of diverse Pareto solution.                                     | Any                | N    | NeurIPS 2021 | $O(m^2 n K^2 )$ |
    | [MGDA](https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf) [code](https://github.com/intel-isl/MultiObjectiveOptimization) | Arbitray Pareto solutions. Location affected highly by initialization. | Any                | Y    | NeurIPS 2018 | $O(m^2 n K )$   |
    | [PMTL](https://proceedings.neurips.cc/paper_files/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf) [code](https://github.com/Xi-L/ParetoMTL)                             | Pareto solutions in sectors.                                          | 2. 3 is difficult. | N    | NeurIPS 2019 | $O(m^2 n K^2 )$ |
    | PMGDA     | Pareto solutions satisfying any preference.                           | Any                | N    | Under review | $O(m^2 n K )$   |
    | [GradienHV](https://arxiv.org/abs/2102.04523) [WangHao](https://link.springer.com/chapter/10.1007/978-3-319-54157-0_44) [code](https://github.com/timodeist/multi_objective_learning) | It is a gradient-based HV method.| 2/3                | N    | CEC 2023| $O(m^2 n K^2 )$ |   
    | Aggregation fun. based, e.g. Tche,mTche,LS,PBI,...                                                                                                                                  | Pareto solution with aggregations.    | Any                | Y    |


    Here, $m$ is the number of objectives, $K$ is the number of samples, and $n$ is the number of decision variables.
    For neural network based methods, $n$ is the number of parameters; hence $n$ is very large (>10000), K is also large (e.g., 20-50), while $m$ is small (2.g., 2-4).

    As a result, m^2 is not a big problem. n^2 is a big problem. K^2 is a big problem.

    For running time consideration, .
        -1 T1. 

    MOO-SVGD is the slowest one.


    EPO, MOO-SVGD, PMTL, 

    
    Current support:
        GradAggSolver, MGDASolver, EPOSolver, MOOSVGDSolver, GradHVSolver, PMTLSolver.



- **PSL solvers**
    -EPO-based
    -Agg-based


- **MOEAs**

-  -- archiving and logging.

- ML pretrained methods. 