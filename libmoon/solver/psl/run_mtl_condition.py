import torch
import argparse
import numpy as np
# from util_global.constant import get_device
from libmoon.problem.mtl.objectives import from_name
from libmoon.util_global.constant import get_device

loss_func_arr = from_name( names=['CrossEntropyLoss', 'CrossEntropyLoss'], task_names=['l', 'r'] )
from model.mtl import HyperNet, LeNetTarget
from libmoon.problem.mtl.loaders import MultiMNISTData
from tqdm import tqdm

from matplotlib import pyplot as plt



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--n-epoch', type=int, default=5)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--n-obj', type=int, default=2)
    parse.add_argument('--model', type=str, default='lenet')
    parse.add_argument('--ray-hidden', type=int, default=100)

    dataset = MultiMNISTData('mnist', 'train')
    args = parse.parse_args()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)
    dataset_test = MultiMNISTData('mnist', 'test')
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)

    args.device = get_device()
    if args.model == 'lenet':
        hnet = HyperNet( [9,5] )
        net = LeNetTarget( [9,5] )
    else:
        assert False, 'model not supported'

    hnet.to(args.device)
    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr)



    loss_history = []

    for idx in tqdm(range( args.n_epoch)):
        for i, batch in enumerate(loader):

            ray = torch.from_numpy(
                np.random.dirichlet((1, 1), 1).astype(np.float32).flatten()
            ).to(args.device)  # ray.shape (1,2)

            batch_ = {}
            for k, v in batch.items():
                batch_[k] = v.to(args.device)

            hnet.train()
            weights = hnet( ray )
            logits_l, logits_r = net(batch_['data'], weights)

            batch_['logits_l'] = logits_l
            batch_['logits_r'] = logits_r

            loss_arr = torch.stack([loss(**batch_) for loss in loss_func_arr])
            # print()
            optimizer.zero_grad()
            loss = ray@loss_arr
            (loss).backward()
            loss_item = loss.cpu().detach().numpy()
            loss_history.append(loss_item)

            optimizer.step()


    plt.plot( np.log( np.array(loss_history) )  )
    plt.show()


