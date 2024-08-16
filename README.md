# LibMoon: A Gradient-based MultiObjective OptimizatioN Library in PyTorch

[![Documentation Status](https://readthedocs.org/projects/libmtl/badge/?version=latest)](https://libmoondocs.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/xzhang2523/libmoon/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/LibMTL.svg)](https://badge.fury.io/py/LibMTL)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/LibMTL.svg?logo=python&logoColor=FFE873)](https://github.com/xzhang2523/libmoon)
[![CodeFactor](https://www.codefactor.io/repository/github/median-research-group/libmtl/badge/main)](https://www.codefactor.io/repository/github/median-research-group/libmtl/overview/main)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxzhang2523%2Flibmoon&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Made With Friends](https://img.shields.io/badge/Made%20With-Love-orange.svg)](https://github.com/xzhang2523/libmoon) 

``LibMoon`` is an open-source library built on [PyTorch](https://pytorch.org/) for gradient-based MultiObjective (MOO). See the [latest documentation](https://libmtl.readthedocs.io/en/latest/) for detailed introductions and API instructions.

Star and fork us on GitHub — it motivates us a lot!

## Main Contributors

| **Name**           | **Institution** | **Role**                              |
|--------------------|-----------------|---------------------------------------|
| **Xiaoyuan Zhang** | CityUHK         | Pareto set learning, gradient-based solver, project lead |
| **Liang Zhao**     | CityUHK         | MOBO                                  |
| **Yingying Yu**    | CityUHK         | Software design                       |
| **Xi Lin**         | CityUHK         | Software design                       |

The first three authors contribute equally to this work.

## Advisory Board
| **Name**               | **Affiliation**                                 | **Role**                                            |
|------------------------|-------------------------------------------------|-----------------------------------------------------|
| **Xuehai Pan**         | PKU                                             | For software design                                 |
| **Hongzong Li**        | CityUHK                                         | For local search                                    |
| **Prof. Jingda Deng**  | Xi'an Jiaotong University                       | For advice on High-D hypervolume computation        |
| **Prof. Yifan Chen**   | Hong Kong Baptist University                    | For advice on OR                                    |
| **Prof. Ke Shang**     | Shenzhen University                             | For advice on approximate hypervolume-based methods |
| **Prof. Genghui Li**   | Shenzhen University                             | For advice on MOEAs                                 |
| **Prof. Han Zhao**     | University of Illinois at Urbana-Champaign      | For advice on fairness classification               |
| **Prof. Zhenkun Wang** | SusTech                                         | Paper writing and maintenance of community          |
| **Prof. Tao Qin**      | MSRA                                            | Industry application                                |
| **Prof. Qingfu Zhang** | CityUHK                                         | Corresponding                                       |


## Contact
- Xiaoyuan Zhang [xzhang2523-c@my.cityu.edu.hk]
- QQ group:


 <p align="center">
  <img src="img/qq.jpg" alt="Moon" width="200">
</p>

- Slack: https://cityu-hiv5987.slack.com/archives/C07GU3N2H2S

### Synthetic Problems

Here's a list of synthetic problems along with relevant research papers and project/code links:

| Problem | Paper                                                                | Project/Code                                         |
|---------|----------------------------------------------------------------------|------------------------------------------------------|
| ZDT     | [Paper](https://ieeexplore.ieee.org/document/996017)                 | [Project](https://pymoo.org/problems/multi/zdt.html) |
| DTLZ    | [Paper](https://ieeexplore.ieee.org/document/996017)                 | [Project](https://pymoo.org/problems/many/dtlz.html) |
| MAF     | [Paper](https://link.springer.com/article/10.1007/s40747-017-0039-7) | [Project](https://pymoo.org/problems/multi/maf.html) |
| WFG     | [Paper](https://ieeexplore.ieee.org/document/996017)                 | [Code](https://github.com/sample-repo/wfg-code)      |
| Fi's    | [Paper](https://ieeexplore.ieee.org/document/996017)                 | [Code](https://github.com/sample-repo/fis-code)      |
| RE      | [Paper](https://arxiv.org/abs/2009.12867)                            | [Code](https://github.com/ryojitanabe/reproblems)    |

### Multitask Learning Problems

This section details problems related to multitask learning, along with their corresponding papers and project/code
references:

| Problem                 | Paper                                                                                                                | Project/Code                                           |
|-------------------------|----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| MO-MNISTs               | [PMTL](https://proceedings.neurips.cc/paper_files/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf)        | [COSMOS](https://github.com/ruchtem/cosmos)            |
| Fairness Classification | [COSMOS](https://arxiv.org/pdf/2103.13392.pdf)                                                                       | [COSMOS](https://github.com/ruchtem/cosmos)            |
| Federated Learning      | [Federal MTL](https://proceedings.neurips.cc/paper_files/paper/2023/file/7cb2c2a8d35576c00078b6591ec26a7d-Paper.pdf) | [COSMOS](https://github.com/ruchtem/cosmos)            |
| Synthetic (DST, FTS...) | [Envelop](https://proceedings.neurips.cc/paper_files/paper/2019/file/4a46fbfca3f1465a27b210f4bdfe6ab3-Paper.pdf)     | [Project](https://github.com/sample-repo/envelop-code) |
| Robotics (MO-MuJoCo...) | [PGMORL](http://proceedings.mlr.press/v119/xu20h/xu20h.pdf)                                                          | [Code](https://github.com/mit-gfx/PGMORL)              |

## Current Supported Solvers

LibMOON includes a variety of solvers tailored for different needs as img below shows. The following solvers are
currently:

<img src="img/supported_methods.png" width="500"></img>

## Gradient-based MOO Solver
  - GradAggSolver
  - EPOSolver
  - MOO-SVGDSolver (*)
  - MGDASolver
  - PMGDASolver
  - PMTLSolver
  - HVGradSolver

(*) The original MOO-SVGD code does not include an implementation for Multitask Learning (MTL). Our release of MOO-SVGD
is the first open-source code that supports MTL.

| Method                                                                                                                                                                             | Property                                                               | #Obj               | Support | Published    | Complexity      |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|--------------------|---------|--------------|-----------------|
| Aggregation fun. based, e.g. Tche,mTche,LS,PBI,...                                                                                                                                 | Pareto solution with aggregations.                                     | Any                | Y       |              |                 |
| [COSMOS](https://arxiv.org/pdf/2103.13392.pdf) [code](https://github.com/ruchtem/cosmos)                                                                                           | Approximated exact solution.                                           | Any                | Y       | ICDM 2021    | $O(m n K )$     |
| [EPO](https://proceedings.mlr.press/v119/mahapatra20a/mahapatra20a.pdf) [code](https://github.com/dbmptr/EPOSearch)                                                                | Exact solution.                                                        | Any                | Y       | ICML 2020    | $O(m^2 n K )$   |
| [MOO-SVGD](https://openreview.net/pdf?id=S2-j0ZegyrE) [code](https://github.com/gnobitab/MultiObjectiveSampling)                                                                   | A set of diverse Pareto solution.                                      | Any                | Y       | NeurIPS 2021 | $O(m^2 n K^2 )$ |
| [MGDA](https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf) [code](https://github.com/intel-isl/MultiObjectiveOptimization)                  | Arbitray Pareto solutions. Location affected highly by initialization. | Any                | Y       | NeurIPS 2018 | $O(m^2 n K )$   |
| [PMGDA](http://arxiv.org/abs/2402.09492)                                                                                                                                           | Pareto solutions satisfying any preference.                            | Any                | Y       | Under review | $O(m^2 n K )$   |
| [PMTL](https://proceedings.neurips.cc/paper_files/paper/2019/file/685bfde03eb646c27ed565881917c71c-Paper.pdf) [code](https://github.com/Xi-L/ParetoMTL)                            | Pareto solutions in sectors.                                           | 2. 3 is difficult. | Y       | NeurIPS 2019 | $O(m^2 n K^2 )$ |
| [HVGrad](https://arxiv.org/abs/2102.04523) [WangHao](https://link.springer.com/chapter/10.1007/978-3-319-54157-0_44) [code](https://github.com/timodeist/multi_objective_learning) | It is a gradient-based HV method.                                      | 2/3                | Y       | CEC 2023     | $O(m^2 n K^2 )$ |   

Here, $m$ is the number of objectives, $K$ is the number of samples, and $n$ is the number of decision variables.
For neural network based methods, $n$ is the number of parameters; hence $n$ is very large (>10000), $K$ is also large (
e.g., 20-50), while $m$ is small (2.g., 2-4).
As a result, $m^2$ is not a big problem. $n^2$ is a big problem. $K^2$ is a big problem.

Time complexity of gradient based methods are as follows,

1. Tier 1. GradAggSolver.
2. Tier 2. MGDASolver, EPOSolver, PMTLSolver.
3. Tier 3. GradHVSolver
4. Tier 4. MOOSVGDSolver

Important things to notice:
The original code MOO-SVGD does not offer a MTL implement. Our code is the first open source code for MTL MOO-SVGD.

## Pareto set learning(PSL) Solvers

LibMOON supports various models of PSL solvers, categorized as follows:

- EPO-based PSL
- Agg-based PSL
- PMGDA-based PSL
- Evolutionary-based PSL

## MultiObjective Bayesian Optimization (MOBO) Solvers

- PSL-MONO
- PSL-DirHV-EI
- DirHV-EGO

## Installation

Libmoon is available on PyPI. You can install it using pip:

```bash
pip install libmoon==0.1.11
```

- **Example1**: Finding a size-K (K=5) Pareto solutions with four lines of code.

```python
from libmoon.solver.gradient.methods import EPOSolver
from libmoon.util_global.initialization import synthetic_init
from libmoon.util_global.weight_factor import uniform_pref
from libmoon.util_global import get_problem

problem = get_problem(problem_name='ZDT1')
prefs = uniform_pref(n_prob=5, n_obj=problem.n_obj, clip_eps=1e-2)
solver = EPOSolver(problem, step_size=1e-2, n_iter=1000, tol=1e-2)
res = solver.solve(x=synthetic_init(problem, prefs), prefs=prefs)
```

- **Example2**: PSL in a problem with three lines of solving problem and two lines of
  evaluating the results.

```python
from libmoon.solver.psl.core_psl import AggPSLSolver
from libmoon.util_global import get_problem
from libmoon.util_global.weight_factor import uniform_pref
from torch import Tensor

problem = get_problem(problem_name='ZDT1')
# agg list [ ’ls ’, ’tche ’, ’mtche ’, ’pbi ’, ... ]
prefs = uniform_pref(n_prob=100, n_obj=problem.n_obj, clip_eps=1e-2)
solver = AggPSLSolver(problem, agg='ls')
model = solver.solve()
eval_y = problem.evaluate(model(Tensor(prefs).cuda()))

```

If you find our code useful, please cite our paper:

```bibtex
@software{libmoon_2024,
  author = {Zhang, Xiaoyuan and Zhao, Liang and Yu, Yingying and Lin, Xi and Chen, Yifan and Zhao, Han and Zhang, Qingfu},
  title = {{LibMOON: A Gradient-based MultiObjective
OptimizatioN Library in PyTorch}},
  url = {https://github.com/xzhang2523/libmoon},
  version = {2.0.4},
  year = {2024}
}
```
