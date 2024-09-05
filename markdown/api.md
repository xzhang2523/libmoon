### API Reference: Adding a New Architecture

#### Class: `SimplePSLLoRAModel`
*Inherits from:* `SimplePSLModel`

**Initialization**
```python
def __init__(self, n_obj, n_var, lr=1e-3):
```
- **Parameters:**
  - `n_obj` (int): Number of objectives.
  - `n_var` (int): Number of variables.
  - `lr` (float, optional): Learning rate. Default is `1e-3`.

**Forward Method**
```python
def forward(self, prefs):
```
- **Input:**
  - `prefs` (Tensor): Preference matrix of shape `(n_prob, n_obj)`.
- **Output:**
  - `solution` (Tensor): Solution matrix of shape `(n_prob, n_var)`.

**Optimization Method**
```python
def optimize(self, problem, epoch):
```
- **Input:**
  - `problem` (ProblemClass): The problem class instance to optimize.
  - `epoch` (int): Number of epochs for optimization.
- **Output:**
  - `None`

**Evaluation Method**
```python
def evaluate(self, prefs):
```
- **Input:**
  - `prefs` (Tensor): Preference matrix of shape `(n_prob, n_obj)`.
- **Output:**
  - `decision_variables` (Tensor): Decision variables of shape `(n_prob, n_var)`.