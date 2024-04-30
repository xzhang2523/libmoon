### Moon: A Standardized/Flexible Framework for MultiObjective OptimizatioN


<img src="moon.png" alt="Moon" width="200">

# Moon: A Multiobjective Optimization Framework

## Introduction
**Moon** is a multiobjective optimization framework that spans from single-objective optimization to multiobjective optimization. It aims to enhance the understanding of optimization problems and facilitate fair comparisons between MOO algorithms.

> "I raise my cup to invite the moon.  
> With my shadow we become three from one."  
> -- Li Bai

## Main Contributors
- **Xiaoyuan Zhang** (Project Leader)
- Ji Cheng
- Liao Zhao
- Weiduo Liao
- Zhe Zhao
- Xi Lin
- Cheng Gong
- Longcan Chen
- YingYing Yu

## Advisory Board
- **Prof. Yifan Chen** (Hong Kong Baptist University)
- **Prof. Zhichao Lu** (City University of Hong Kong)
- **Prof. Ke Shang** (Shenzhen University)
- **Prof. Tao Qin** (Microsoft Research)
- **Prof. Han Zhao** (University of Illinois at Urbana-Champaign)

## Correspondence
For any inquiries, please contact **Prof. Qingfu Zhang** (City University of Hong Kong) at the corresponding address.

## Resources
For more information on methodologies, please visit our [GitHub repository](https://github.com/xzhang2523/awesome-moo-ml-papers). Contributions and stars are welcome!



(1) A standardlized gradient based framework. 
# Optimization Problem Classes

## **Problem Class Details**
For more information on problem specifics, please refer to the `Readme_problem.md` file.

### Synthetic Problems
Here's a list of synthetic problems along with relevant research papers and project/code links:

| Problem | Paper | Project/Code |
|---------|-------|--------------|
| ZDT     | [Paper](https://ieeexplore.ieee.org/document/996017) | [Project](https://pymoo.org/problems/multi/zdt.html) |
| DTLZ    | [Paper](https://ieeexplore.ieee.org/document/996017) | [Project](https://pymoo.org/problems/many/dtlz.html) |
| MAF     | [Paper](https://link.springer.com/article/10.1007/s40747-017-0039-7) | [Project](https://pymoo.org/problems/multi/maf.html) |
| WFG     | [Paper](https://ieeexplore.ieee.org/document/996017) | [Code](https://github.com/sample-repo/wfg-code) |
| Fi's    | [Paper](https://ieeexplore.ieee.org/document/996017) | [Code](https://github.com/sample-repo/fis-code) |
| RE      | [Paper](https://arxiv.org/abs/2009.12867) | [Code](https://github.com/ryojitanabe/reproblems) |

### Multitask Learning Problems

This section details problems related to multitask learning, along with their corresponding papers and project/code references:

| Problem              | Paper                                                                                                           | Project/Code                                   |
|----------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| MO-MNISTs            | [PMTL](https://proceedings.neurips.cc/paper_files/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf) | [COSMOS](https://github.com/ruchtem/cosmos)    |
| Fairness Classification | [COSMOS](https://arxiv.org/pdf/2103.13392.pdf)                                                                  | [COSMOS](https://github.com/ruchtem/cosmos)    |
| Federated Learning   | [Federal MTL](https://proceedings.neurips.cc/paper_files/paper/2023/file/7cb2c2a8d35576c00078b6591ec26a7d-Paper.pdf) | [COSMOS](https://github.com/ruchtem/cosmos) |
| Synthetic (DST, FTS...) | [Envelop](https://proceedings.neurips.cc/paper_files/paper/2019/file/4a46fbfca3f1465a27b210f4bdfe6ab3-Paper.pdf) | [Project](https://github.com/sample-repo/envelop-code) |
| Robotics (MO-MuJoCo...) | [PGMORL](http://proceedings.mlr.press/v119/xu20h/xu20h.pdf)                                                      | [Code](https://github.com/mit-gfx/PGMORL)     |



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

    Time complexity of gradient based methods are as follows,
        -1 Tier 1. GradAggSolver.
        -2 Tier 2. MGDASolver, EPOSolver, PMTLSolver. 
        -3 Tier 3. GradHVSolver
        -4 Tier 4. MOOSVGDSolver

    Current support:
        GradAggSolver, MGDASolver, EPOSolver, MOOSVGDSolver, GradHVSolver, PMTLSolver.

    Important things to notice:
        The original code MOO-SVGD does not offer a MTL implement. Our code is the first open source code for MTL MOO-SVGD.

## Supported Solvers

### Current Support
Libmoon includes a variety of solvers tailored for different needs:
- GradAggSolver
- MGDASolver
- EPOSolver
- MOOSVGDSolver (*)
- GradHVSolver
- PMTLSolver

(*) The original MOO-SVGD code does not include an implementation for Multitask Learning (MTL). Our release of MOO-SVGD is the first open-source code that supports MTL.

## PSL (Pareto set learning) Solvers

Libmoon supports various models of PSL solvers, categorized as follows:
- EPO-based PSL model
- Agg-based PSL model 
- Hypernetwork-based PSL model 
- ConditionalNet-based PSL model 
- Simple PSL model
- Generative PSL model




## MOEA/D Framework

### Currently Supported
- Vanilla [MOEA/D](https://ieeexplore.ieee.org/document/4358754)

### Upcoming Releases
- [MOEA/D AWA](https://pubmed.ncbi.nlm.nih.gov/23777254/)
- [MOEA/D Neural AWA](https://openreview.net/pdf?id=W3T9rql5eo)

## ML Pretrained Methods
- HV Net, a model for handling high-volume data, available [here](https://arxiv.org/abs/2203.02185).

## Installation

Libmoon is available on PyPI. You can install it using pip:

```bash
pip install libmoon==0.1.11


Example code for a synthetic problem,
```
from libmoon.solver.gradient import GradAggSolver
from libmoon.util_global.constant import problem_dict
from libmoon.util_global.weight_factor.funs import uniform_pref
import torch
import numpy as np
from matplotlib import pyplot as plt
import argparse
from libmoon.visulization.view_res import vedio_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument('--n-partition', type=int, default=10)
    parser.add_argument('--agg', type=str, default='tche')  # If solve is agg, then choose a specific agg method.
    parser.add_argument('--solver', type=str, default='agg')
    parser.add_argument('--problem-name', type=str, default='VLMOP2')
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--step-size', type=float, default=1e-2)
    parser.add_argument('--tol', type=float, default=1e-6)
    args = parser.parse_args()
    
    
    # Init the solver, problem and prefs. 
    solver = GradAggSolver(args.step_size, args.iter, args.tol)
    problem = problem_dict[args.problem_name]
    prefs = uniform_pref(args.n_partition, problem.n_obj, clip_eps=1e-2)
    args.n_prob = len(prefs)

    # Initialize the initial solution 
    if 'lbound' in dir(problem):
        if args.problem_name == 'VLMOP1':
            x0 = torch.rand(args.n_prob, problem.n_var) * 2 / np.sqrt(problem.n_var) - 1 / np.sqrt(problem.n_var)
        else:
            x0 = torch.rand(args.n_prob, problem.n_var)
    else:
        x0 = torch.rand( args.n_prob, problem.n_var )*20 - 10


    # Solve results
    res = solver.solve(problem, x=x0, prefs=prefs, args=args)
    
    # Visualize results
    y_arr = res['y']
    plt.scatter(y_arr[:,0], y_arr[:,1], s=50)
    plt.xlabel('$f_1$', fontsize=20)
    plt.ylabel('$f_2$', fontsize=20)
    plt.show()
    
    # If use vedio
    use_vedio=True
    if use_vedio:
        vedio_res(res, problem, prefs, args)     
```
        
Example of MTL
```


```

    

