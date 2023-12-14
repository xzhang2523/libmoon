import torch
from torch import nn
import argparse
import numpy as np
# from util_global.constant import get_device
from problem.mtl.objectives import from_name
from util_global.constant import get_device
loss_arr = from_name( names=['CrossEntropyLoss', 'CrossEntropyLoss'], task_names=['l', 'r'] )

from model.mtl import HyperNet, LeNetTarget
from problem.mtl.loaders import MultiMNISTData



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--n-epoch', type=int, default=100)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--n-obj', type=int, default=2)
    parse.add_argument('--model', type=str, default='lenet')
    dataset = MultiMNISTData('mnist', 'train')
    args = parse.parse_args()

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)

    dataset_test = MultiMNISTData('mnist', 'test')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)

    args.device = get_device()
    if args.model == 'lenet':
        hypernet = HyperNet([9,5])
        net = LeNetTarget([9,5])
    else:
        assert False, 'model not supported'


    hypernet.to(args.device)
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=args.lr)

    for idx in range( args.n_epoch ):
        for i, batch in enumerate(loader):
            prefs = torch.Tensor( np.random.dirichlet(np.ones(args.n_obj), args.batch_size) ).to(args.device)
            hypernet.train()
            weights = hypernet(prefs)

            logits1, logits2 = net(batch['data'], weights)
            print()

    print('hello world')