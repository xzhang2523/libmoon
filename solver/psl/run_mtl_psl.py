import torch
from torch import nn
import argparse
import numpy as np
# from util_global.constant import get_device
from problem.mtl.objectives import from_name
from util_global.constant import get_device
loss_arr = from_name( names=['CrossEntropyLoss', 'CrossEntropyLoss'], task_names=['l', 'r'] )

from model.mtl import HyperNet

# loss_1 = CrossEntropyLoss(label_name='labels_l', logits_name='logits_l')
# loss_2 = CrossEntropyLoss(label_name='labels_r', logits_name='logits_r')





if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--n-epoch', type=int, default=100)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--n-obj', type=int, default=2)

    args = parse.parse_args()
    args.device = get_device()


    hypernet = HyperNet()
    hypernet.to(args.device)
    optimizer = torch.optim.Adam(hypernet.parameters(), lr=args.lr)

    for idx in range( args.n_epoch ):
        prefs = torch.Tensor( np.random.dirichlet(np.ones(args.n_obj), args.batch_size) ).to(args.device)
        hypernet.train()
        weights = hypernet(prefs)
        logits1, logits2 = net(image, weights)




    print('hello world')