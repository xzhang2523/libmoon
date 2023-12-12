import torch
from torch import nn
import argparse
# We will still first use MO-MNIST as an example first.
import numpy as np
from util.constant import get_device




if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--n-epoch', type=int, default=100)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--n-obj', type=int, default=2)

    args = parse.parse_args()
    args.device = get_device()




    for idx in range(args.n_obj):
        prefs = torch.Tensor( np.random.dirichlet(np.ones(args.n_obj), args.batch_size) ).to(args.device)





    print('hello world')