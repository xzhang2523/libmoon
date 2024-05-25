
import torch
import numpy as np
from tqdm import tqdm
from libmoon.solver.psl.model import SimplePSLModel


from torch.autograd import Variable
from libmoon.util_global import get_problem




class AggPSL:
    def __init__(self, problem, batch_size, device, lr=1e-3, epoch=1000, agg='ls', use_es=False):

        self.problem = problem
        self.batch_size = batch_size
        self.device = device
        self.epoch = epoch
        self.model = SimplePSLModel(problem).to(self.device)
        self.use_es = use_es
        self.agg = agg
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr=lr )



    def solve(self):
        n_obj = self.problem.n_obj

        for _ in tqdm( range(self.epoch) ):
            prefs = torch.Tensor(np.random.dirichlet(np.ones(n_obj),
                                                     self.batch_size)).to(self.device)
            xs = self.model(prefs)
            fs = self.problem.evaluate(xs)

            if self.use_es:
                # Part 1. Estimating term A.
                agg_func = agg_dict[args.agg]
                fs_var = Variable(fs, requires_grad=True)
                g = agg_func(fs_var, prefs)
                loss_g = torch.mean(g)
                loss_history.append(loss_g.cpu().detach().numpy())
                g.sum().backward()
                termA = (fs_var.grad).unsqueeze(1)
                # Part 2. Estimating term B.
                termB = ES_gradient_estimation_batch(problem, xs.cpu().detach().numpy())
                termB = torch.Tensor(termB).to(args.device)
                xs = model(prefs)
                res = termA @ termB
                loss = torch.mean(res @ xs.unsqueeze(2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                agg_func = agg_dict[args.agg]
                g = agg_func(fs, prefs)
                loss = torch.mean(g)
                loss_history.append(loss.cpu().detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model





if __name__ == '__main__':
    problem = get_problem(problem_name='DTLZ2', n_var=30)

    psler = AggPSL(problem, batch_size=128, device='cuda', lr=1e-3, epoch=1000, agg='ls', use_es=False)



