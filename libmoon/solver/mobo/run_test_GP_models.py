import torch
import math
import numpy as np  
from utils import lhs
from surrogate_models import GaussianProcess
from matplotlib import pyplot as plt

# TODO: building GP models using Gpytorch 
  
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
n_init, n_test = 5, 100
n_var, n_obj = 1, 2
 

train_x = torch.from_numpy(lhs(n_var, samples=n_init)).to(**tkwargs) 
 
train_y = torch.stack([
    torch.sin(12*train_x -4)*torch.pow(6*train_x-2,2),
    torch.cos(train_x * (2 * math.pi)), 
])[:,:,0].T
 
test_x = torch.linspace(0, 1, n_test).reshape(n_test, n_var)
 
test_x.requires_grad = True
test_y = torch.stack([
    torch.sin(12*test_x -4)*torch.pow(6*test_x-2,2),
    torch.cos(test_x * (2 * math.pi)),
])[:,:,0].T

train_x_np = train_x.detach().cpu().numpy()
train_y_np = train_y.detach().cpu().numpy()
test_x_np = test_x.detach().cpu().numpy()
test_y_np =  test_y.detach().cpu().numpy()
 

# GP modeling via Sklearn  
gp_skl =  GaussianProcess(n_var, n_obj)
gp_skl.fit(train_x_np, train_y_np) 
 
out = gp_skl.evaluate(test_x_np, std=True, calc_gradient=True) 
mean, std, mean_grad, std_grad = out['F'], out['S'], out['dF'], out['dS']
 
# plot 
upper = mean + std
lower = mean - std
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
# Plot training data as black stars
y1_ax.plot(train_x_np, train_y_np[:, 0], 'k*', label="observed samples")
# Predictive mean as blue line
y1_ax.plot(test_x_np, mean[:, 0], color='blue', linewidth=2.0, linestyle="-", label='$\mu_1(x)$')
y1_ax.plot(test_x_np, test_y_np[:,0], color='black', linewidth=2.0, linestyle="--", label='$f_1(x)=\sin(2\pi x)$')
# Shade in confidence
y1_ax.fill_between(test_x_np[:,0], lower[:, 0], upper[:, 0], alpha=0.5, label='$\mu_1(x)\pm \sigma_1(x)$')
# y1_ax.set_ylim([-3, 3])
y1_ax.legend(loc='upper left')
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
y2_ax.plot(train_x_np, train_y_np[:, 1], 'k*', label="observed samples")
# Predictive mean as blue line
y2_ax.plot(test_x_np, mean[:, 1], color='blue', linewidth=2.0, linestyle="-", label='$\mu_2(x)$')
y2_ax.plot(test_x_np, test_y_np[:,1], color='black', linewidth=2.0, linestyle="--", label='$f_2(x)=\cos(2\pi x)$')
# Shade in confidence
y2_ax.fill_between(test_x_np[:,0], lower[:, 1], upper[:, 1], alpha=0.5, label='$\mu_2(x)\pm \sigma_2(x)$')
# y2_ax.set_ylim([-3, 3])
y2_ax.legend(loc='upper left')
y2_ax.set_title('Observed Values (Likelihood)')
 
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
