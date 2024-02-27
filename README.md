### Moon: A Standardized/Flexible Framework for MultiObjective OptimizatioN
<img src="moon.png" alt="Moon" width="200">

''
    I raise my cup to invite the moon.
    With my shadow we become three from one.
''
-- Li Bai.

Moon: is a multiobjective optimization framework, from single-objective optimization to multiobjective optimization, towards a better understanding of optimization problems.


Do not release.  

Main contributors: Xiaoyuan Zhang, Ji Cheng, Liao Zhang, Weiduo Liao, Xi Lin.

Corresponding to: Prof. Qingfu Zhang. 

Advised by: Prof. Yifan Chen, Prof. Zhichao Lu, Prof. Ke Shang, Prof. Tao Qin. 

This project has four important parts:

(1) A standardlized gradient based framework. 

- **Problem** class. For more problem details, please also check the Readme_problem.md file. 
  (1) For synthetic problems,
- 
  | Problem                                                      | Paper                                                                | Project/Code                                         |
  |--------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------|
  | ZDT                                                          | [paper](https://ieeexplore.ieee.org/document/996017)                 | [project](https://pymoo.org/problems/multi/zdt.html) |
  | DTLZ                                                         | [paper] | [project](https://pymoo.org/problems/many/dtlz.html) |Y                                        |
  | MAF                                                          | [paper](https://link.springer.com/article/10.1007/s40747-017-0039-7) | [project]                         |
  | [WFG](https://ieeexplore.ieee.org/document/996017) [code]()  | Real world problems.                                                 | Y                                                    |
  | [Fi's](https://ieeexplore.ieee.org/document/996017) [code]() | Real world problems.                                                 | Y                                                    |
  | RE                                                           | [paper](https://arxiv.org/abs/2009.12867)                            | [code](https://github.com/ryojitanabe/reproblems)    |

(2) For multitask learning problems, 

| Problem                 | Paper | Project/Code |
|-------------------------|------|--------------|
| MO-MNISTs               | [PMTL](https://proceedings.neurips.cc/paper_files/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf)     | [COSMOS](https://github.com/ruchtem/cosmos)     |
| Fairness Classification |[COSMOS](https://arxiv.org/pdf/2103.13392.pdf) |[COSMOS](https://github.com/ruchtem/cosmos) |
| Federated Learning      | | |

(3) For MORL problems,

| Problem                 | Paper                                                                                                            | Project/Code                              |
|-------------------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| Synthetic (DST FTS...)  | [Envelop](https://proceedings.neurips.cc/paper_files/paper/2019/file/4a46fbfca3f1465a27b210f4bdfe6ab3-Paper.pdf) | [code]()                                  |
| Robotics (MO-MuJoCo...) | [PGMORL](http://proceedings.mlr.press/v119/xu20h/xu20h.pdf)                                                      | [code](https://github.com/mit-gfx/PGMORL) |




- **Gradient-based Solver**.

    | Method                                                                                                                                                                                | Property                                                              | #Obj               | Support | Published | Complexity      |
    |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------|---------|----------|-----------------|
    | [EPO](https://proceedings.mlr.press/v119/mahapatra20a/mahapatra20a.pdf) [code](https://github.com/dbmptr/EPOSearch)                                                                   | Exact solution.                                                       | Any                | Y       | ICML 2020 | $O(m^2 n K )$   |
    | [COSMOS](https://arxiv.org/pdf/2103.13392.pdf) [code](https://github.com/ruchtem/cosmos)                                                                                              | Approximated exact solution.                                          | Any                | Y       | ICDM 2021| $O(m n K )$     |
    | [MOO-SVGD](https://openreview.net/pdf?id=S2-j0ZegyrE) [code](https://github.com/gnobitab/MultiObjectiveSampling)                                                                      | A set of diverse Pareto solution.                                     | Any                | Y       | NeurIPS 2021 | $O(m^2 n K^2 )$ |
    | [MGDA](https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf) [code](https://github.com/intel-isl/MultiObjectiveOptimization)                     | Arbitray Pareto solutions. Location affected highly by initialization. | Any                | Y       | NeurIPS 2018 | $O(m^2 n K )$   |
    | [PMTL](https://proceedings.neurips.cc/paper_files/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf) [code](https://github.com/Xi-L/ParetoMTL)                               | Pareto solutions in sectors.                                          | 2. 3 is difficult. | Y       | NeurIPS 2019 | $O(m^2 n K^2 )$ |
    | [PMGDA](http://arxiv.org/abs/2402.09492)                                                                                                                                              | Pareto solutions satisfying any preference.                           | Any                | Y       | Under review | $O(m^2 n K )$   |
    | [GradienHV](https://arxiv.org/abs/2102.04523) [WangHao](https://link.springer.com/chapter/10.1007/978-3-319-54157-0_44) [code](https://github.com/timodeist/multi_objective_learning) | It is a gradient-based HV method.| 2/3                | Y       | CEC 2023| $O(m^2 n K^2 )$ |   
    | Aggregation fun. based, e.g. Tche,mTche,LS,PBI,...                                                                                                                                    | Pareto solution with aggregations.    | Any                | Y       |


    Here, $m$ is the number of objectives, $K$ is the number of samples, and $n$ is the number of decision variables.
    For neural network based methods, $n$ is the number of parameters; hence $n$ is very large (>10000), K is also large (e.g., 20-50), while $m$ is small (2.g., 2-4).

    As a result, m^2 is not a big problem. n^2 is a big problem. K^2 is a big problem.

    For running time consideration, .
        -1 T1. 

    MOO-SVGD is the slowest one.


    EPO, MOO-SVGD, PMTL, 

    
    Current support:
        GradAggSolver, MGDASolver, EPOSolver, MOOSVGDSolver, GradHVSolver, PMTLSolver.

    Important things to notice:
        The original code MOO-SVGD does not offer a MTL implement. Our code is the first open source code for MTL MOO-SVGD.


- 
- **PSL solvers**
    - EPO-based
    - Agg-based
    - Hypernetwork-based
    - ConditionalNet-based
    - Simple PSL model
    
    
- **MOEA/D**
    Current supported:
    - Vanilla [MOEA/D](https://ieeexplore.ieee.org/document/4358754)
    
    - Will be released soon:
    - [MOEA/D AWA](https://pubmed.ncbi.nlm.nih.gov/23777254/). 
    - [MOEA/D neural AWA](https://openreview.net/pdf?id=W3T9rql5eo).

- **MOBO**
    - 

- ML pretrained methods. 
    - HV net. 


Examples:
```
from libmoon.solver.gradient import GradAggSolver
