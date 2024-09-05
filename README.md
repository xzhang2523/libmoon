# LibMOON: A Gradient-based MultiObjective OptimizatioN Library in PyTorch


[![Documentation Status](https://readthedocs.org/projects/libmoondocs/badge/?version=latest)](https://libmoondocs.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/xzhang2523/libmoon/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/LibMOON.svg)](https://badge.fury.io/py/LibMOON)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/LibMOON.svg?logo=python&logoColor=FFE873)](https://github.com/xzhang2523/libmoon)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fxzhang2523%2Flibmoon&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Made With Friends](https://img.shields.io/badge/Made%20With-Love-orange.svg)](https://github.com/xzhang2523/libmoon) 

``LibMoon`` is an open-sourc[e library built on [PyTorch](https://pytorch.org/) for gradient-b]()ased MultiObjective (MOO). See the [latest documentation](https://readthedocs.org/projects/libmoondocs/badge/?version=latest) for detailed introductions and API instructions.

Star or fork us on GitHub — it motivates us a lot!

# News
- **[Aug 27 2024]** Added support for [LoRA-PSL](https://arxiv.org/pdf/2407.20734) (ICML 2024). Many thanks to [Weiyu Chen](https://scholar.google.com/citations?user=Zbg7LycAAAAJ&hl=zh-CN) for his contribution.

- **[June 20 2024]** Supports most popular gradient-based methods: MGDAUB, Random, EPO, PMGDA, Aggregation-based methods, PMTL, HVGrad, MOOSVGD ... 

- **[April 20 2024]** Supports [PMTL](https://arxiv.org/abs/1912.12854) (NeurIPS 2019), [HvGrad](https://arxiv.org/abs/2102.04523) (EMO 2022). Many thanks for Dr. [Xi Lin](https://scholar.google.com/citations?user=QB_MUboAAAAJ&hl=en)'s contribution. Thanks for helpful communications from Dr. [Hao Wang](https://scholar.google.com/citations?user=Pz9c6XwAAAAJ&hl=en).  

- **[March 17 2024]** Supports three [MOBO-PSL](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=QB_MUboAAAAJ&citation_for_view=QB_MUboAAAAJ:W7OEmFMy1HYC) methods. Many thanks to [Liang Zhao](https://scholar.google.com.hk/citations?user=DDGCxNkAAAAJ&hl=zh-CN)'s contribution.

- **[March 10 2024]** Supports [hypernetwork-based](https://openreview.net/pdf/9c01e8c47f7e80e87af0175ac2a5e9a356f518bd.pdf) Pareto set learning methods. 


# 1. LibMOON Supported Problems
## 1.1 Synthetic Problems
Here's a list of synthetic problems along with relevant research papers and project/code links:

| Problem | Paper | Project/Code|
|---------|--------|---------|
| ZDT     | [Paper](https://ieeexplore.ieee.org/document/996017)  | [Project](https://pymoo.org/problems/multi/zdt.html) |
| DTLZ    | [Paper](https://ieeexplore.ieee.org/document/996017) | [Project](https://pymoo.org/problems/many/dtlz.html) |
| MAF     | [Paper](https://link.springer.com/article/10.1007/s40747-017-0039-7) | [Project](https://pymoo.org/problems/multi/maf.html) |
| WFG     | [Paper](https://ieeexplore.ieee.org/document/996017)  | [Code](https://github.com/sample-repo/wfg-code) |
| Fi's    | [Paper](https://ieeexplore.ieee.org/document/996017)  | [Code](https://github.com/sample-repo/fis-code) |
| RE      | [Paper](https://arxiv.org/abs/2009.12867) | [Code](https://github.com/ryojitanabe/reproblems) |


## 1.2 Multiobjective Multitask Learning (MO-MTL) Problems

|Method|$L_1$|$L_2$ |
|---------|----|----|
| Fairness classification [1]     | Binary cross entropy    | DEO                     |
| Multiobjective classification [2] | Cross entropy loss BR   | Cross entropy loss UL   |
| MO machine learning [3]         | Mean square loss        | Mean square loss        |
| MO distribution alignment        | Similarity 1            | Similarity 2            |

**Notes:**
- DEO: Difference of Equality of Opportunity.



# 2. LibMOON Supported Solvers

LibMOON includes a variety of solvers tailored for different needs as img below shows. 
<img src="img/frame.jpg" width="1000"></img>

[//]: # (<img src="img/supported_methods.png" width="500"></img>)

## 2.1 Finite solution solvers
  - GradAggSolver
  - EPOSolver
  - MOO-SVGDSolver
  - MGDASolver
  - PMGDASolver
  - PMTLSolver
  - HVGradSolver

| Method | Property  | Paper| Complexity |
|----|----|--------|---------|
| EPO (Mahapatra et al 2020)| Exact solutions| [paper](https://proceedings.mlr.press/v119/mahapatra20a.html)| $O(m^2nK)$ |
| MGDA-UB (Sener et al 2018) | Arbitrary solutions | [paper](https://arxiv.org/abs/1810.04650)| $O(m^2nK)$ |
| PMGDA (Zhang et al 2024)  | Specific solutions | [paper](https://arxiv.org/abs/2402.09492)| $O(m^2nK)$ |
| Random (Lin et al 2021)   | Arbitrary solutions | [paper](https://arxiv.org/abs/2111.10603)| $O(m^2nK)$ |
| MOO-SVGD (Liu et al 2021) | Diverse solutions | [paper](https://papers.nips.cc/paper_files/paper/2021/hash/7bb16972da003e87724f048d76b7e0e1-Abstract.html)| $O(m^2nK^2)$|
| PMTL (Lin et al 2019)     | Sector solutions | [paper](https://arxiv.org/abs/1912.12854)|$O(m^2nK^2)$|
| HVGrad (Deist et al 2021) | Maximal HV solutions | [paper](https://arxiv.org/abs/2102.04523)|$O(m^2nK^2)$|
| Agg-LS (Miettinen et al 1999) | Convex hull solutions | [book](https://link.springer.com/book/10.1007/978-1-4615-5563-6)| $O(mnK)$|
| Agg-Tche (Zhang et al 2007) | Exact solutions | [paper](https://ieeexplore.ieee.org/document/4358754)|$O(mnK)$| 
| Agg-mTche (Ma et al 2017) | Exact solutions | [paper](https://ieeexplore.ieee.org/document/7927726)|$O(mnK)$| 
| Agg-PBI (Zhang et al 2007) | Approximate exact solutions | [paper](https://ieeexplore.ieee.org/document/4358754)|$O(mnK)$|
| Agg-COSMOS (Ruchte et al 2007) | Approximate exact solutions | [paper](https://arxiv.org/abs/2103.13392)|$O(mnK)$|
| Agg-SoftTche (Lin et al 2024) | Fast approximate exact solutions | [paper](https://arxiv.org/abs/2402.19078)|$O(mnK)$|

Notations:
- $m$ is the number of objectives.
- $K$ is the number of subproblems.
- $n$ is the number of decision variables.

In neural network methods, $n$ (number of parameters) is very large (>10,000), $K$ (number of classes) is also large (e.g., 20-50), and $m$ (data features) is relatively small (e.g., 2-4). Consequently, $m^2$ is not a significant issue, but $n^2$ and $K^2$ are major concerns.

## 2.2 Pareto set learning(PSL) Solvers

LibMOON supports various models of PSL solvers, categorized as follows:

| Method| Property                               | Paper                                                                      |
|-----|----------------------------------------|----------------------------------------------------------------------------|
| EPO-based PSL (Navon et al 2021) | Exact solutions                        | [paper](https://openreview.net/pdf/9c01e8c47f7e80e87af0175ac2a5e9a356f518bd.pdf) |
| PMGDA-based PSL (Zhang et al 2024) | Specific solutions                     | [paper](https://arxiv.org/abs/2402.09492)                                  |
| Aggregation-baed PSL (Sener et al 2018) | Minimal aggregation function solutions | [paper](https://openreview.net/pdf/9c01e8c47f7e80e87af0175ac2a5e9a356f518bd.pdf) |
| Evolutionary PSL (Sener et al 2018) | Mitigate local minimal by ES           | [paper](https://arxiv.org/pdf/2310.20426)                                                                  |
| LoRA PSL (Chen et al 2024)     | Light model structure | [paper](https://openreview.net/pdf?id=a2uFstsHPb)|


## 2.3 MultiObjective Bayesian Optimization (MOBO) Solvers

- PSL-MONO
- PSL-DirHV-EI
- DirHV-EGO

## 3. Installation and quick start

Libmoon is available on PyPI. You can install it using pip:

```bash
pip install libmoon==0.1.11
```

- **Example1**: Finding a size-K (K=5) Pareto solutions with four lines of code.

```python
from libmoon.solver.gradient.methods import EPOSolver
from libmoon.util.initialization import synthetic_init
from libmoon.util.weight_factor import uniform_pref
from libmoon.util import get_problem

problem = get_problem(problem_name='ZDT1')
prefs = uniform_pref(n_prob=5, n_obj=problem.n_obj, clip_eps=1e-2)
solver = EPOSolver(problem, step_size=1e-2, n_iter=1000, tol=1e-2)
res = solver.solve(x=synthetic_init(problem, prefs), prefs=prefs)
```

- **Example2**: PSL in a problem with three lines of solving problem and two lines of
  evaluating the results.

```python
from libmoon.solver.psl.core_psl import AggPSLSolver
from libmoon.util import get_problem
from libmoon.util.weight_factor import uniform_pref
from torch import Tensor

problem = get_problem(problem_name='ZDT1')
# agg list [ ’ls ’, ’tche ’, ’mtche ’, ’pbi ’, ... ]
prefs = uniform_pref(n_prob=100, n_obj=problem.n_obj, clip_eps=1e-2)
solver = AggPSLSolver(problem, agg='ls')
model = solver.solve()
eval_y = problem.evaluate(model(Tensor(prefs).cuda()))

```

## 4. Citing LibMOON and Acknowledgements

### 4.1 If you find our code useful, please cite our codebase:

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

## 4.2 Main Contributors
The first three authors contribute equally to this work.

| **Name**           | **Institution** | **Role**                                                    |
|--------------------|-----------------|-------------------------------------------------------------|
| **Xiaoyuan Zhang** | CityUHK         | Pareto set learning/Discrete solutions solvers/Project lead |
| **Liang Zhao**     | CityUHK         | MOBO                                                        |
| **Yingying Yu**    | CityUHK         | Software design                                             |
| **Xi Lin**         | CityUHK         | Software design                                             |


## 4.3 Advisory Board
We sincernely thank the following contributors for their valuable contributions or feedbacks:

| **Name**               | **Institution** | **Role** |
|------------------------|-----|-----|
| **Xuehai Pan**         | PKU | For software design     |
| **Hongzong Li**        | CityUHK | For local search                          |
| **Zhe Zhao**           | CityUHK/USTC | MODL                                      |
| **Meitong Liu**        | HKU| Feedback|
| **Weiduo Liao**        | CityUHK/SusTech        | HV-Net/Paper writing                      |
| **Weiyu Chen**         | HKUST| For LoRA PSL                              |
| **Prof. Jingda Deng**  | Xi'an Jiaotong University | For advice on High-D hypervolume computation |
| **Prof. Yifan Chen**   | Hong Kong Baptist University | For advice on OR                          |
| **Prof. Ke Shang**     | Shenzhen University    | For advice on approximate hypervolume-based methods |
| **Prof. Genghui Li**   | Shenzhen University    | For advice on MOEAs                       |
| **Prof. Han Zhao**     | University of Illinois at Urbana-Champaign | For advice on fairness classification     |
| **Prof. Zhenkun Wang** | SusTech                | Paper writing and maintenance of community|
| **Prof. Tao Qin**      | MSRA  | Industry application |
| **Prof. Qingfu Zhang** | CityUHK | Corresponding author|

## 4.4 Contact
- Xiaoyuan Zhang [xzhang2523-c@my.cityu.edu.hk]
- QQ group:

 <p align="center">
  <img src="img/qq.jpg" alt="Moon" width="200">
</p>

- Wechat group:
<p align="center">
  <img src="img/wechat.jpg" alt="Moon" width="200">
</p>

- Slack group: https://cityu-hiv5987.slack.com/archives/C07GU3N2H2S

## 4.5 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xzhang2523/libmoon&type=Date)](https://star-history.com/#xzhang2523/libmoon&Date)


