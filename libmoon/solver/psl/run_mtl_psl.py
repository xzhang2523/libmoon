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
from libmoon.util_global.constant import get_agg_func
import pickle



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--n-epoch', type=int, default=10)
    parse.add_argument('--batch-size', type=int, default=128)
    parse.add_argument('--lr', type=float, default=1e-3)
    parse.add_argument('--n-obj', type=int, default=2)
    parse.add_argument('--model', type=str, default='lenet')
    parse.add_argument('--dataset', type=str, default='mnist')

    parse.add_argument('--solver', type=str, default='agg')
    parse.add_argument('--agg', type=str, default='ls')

    parse.add_argument('--ray-hidden', type=int, default=100)
    args = parse.parse_args()


    agg_func = get_agg_func(args.agg)
    if args.solver == 'agg':
        args.task_name = '{}_{}'.format(args.solver, args.agg)
    else:
        args.task_name = args.solver

    print('Training...')
    print('Task name: {}'.format(args.task_name))
    print('Dataset:{}'.format(args.dataset))


    dataset = MultiMNISTData(args.dataset, 'train')
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
    for idx in tqdm(range(args.n_epoch)):
        for i, batch in enumerate(loader):
            ray = torch.from_numpy(
                np.random.dirichlet((1, 1), 1).astype(np.float32).flatten()
            ).to(args.device)  # ray.shape (1,2)
            batch_ = {}
            for k, v in batch.items():
                batch_[k] = v.to(args.device)
            hnet.train()
            weights = hnet(ray)
            logits_l, logits_r = net(batch_['data'], weights)
            batch_['logits_l'] = logits_l
            batch_['logits_r'] = logits_r
            loss_arr = torch.stack([loss(**batch_) for loss in loss_func_arr])
            optimizer.zero_grad()

            if args.solver == 'agg':
                loss_arr = torch.atleast_2d(loss_arr)
                ray = torch.atleast_2d(ray)

                loss = agg_func(loss_arr, ray)

            (loss).backward()
            loss_item = loss.cpu().detach().numpy()
            loss_history.append(loss_item)
            optimizer.step()


    from libmoon.util_global.constant import root_name
    import os
    save_dir = os.path.join(root_name, 'output', 'psl', 'mtl', args.dataset, args.task_name)
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure()

    plt.plot(np.array(loss_history))
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Hypernet loss', fontsize=18)
    save_fig_name = os.path.join(save_dir, 'hypernet_loss.pdf')
    plt.savefig( save_fig_name )
    print('save to {}'.format(save_fig_name))

    test_ray_num = 10
    pref1 = np.linspace(0, 1, test_ray_num)
    pref2 = 1 - pref1
    test_ray = torch.Tensor(np.stack([pref1, pref2], axis=1)).to(args.device)



    print('Testing on test set')
    loss_ray = []
    for ray in test_ray:
        loss_arr_epoch = []
        for i, batch in enumerate(loader):
            batch_ = {}
            for k, v in batch.items():
                batch_[k] = v.to(args.device)
            hnet.train()
            weights = hnet(ray)
            logits_l, logits_r = net(batch_['data'], weights)
            batch_['logits_l'] = logits_l
            batch_['logits_r'] = logits_r
            loss_arr = torch.stack( [loss(**batch_) for loss in loss_func_arr] )
            loss_arr_epoch.append(loss_arr)
            optimizer.zero_grad()
            loss = ray @ loss_arr
            (loss).backward()
            loss_item = loss.cpu().detach().numpy()
            loss_history.append(loss_item)
            optimizer.step()
        loss_arr_epoch = torch.stack(loss_arr_epoch)
        loss_arr_epoch_mean = torch.mean(loss_arr_epoch, dim=0)
        loss_ray.append( loss_arr_epoch_mean )

    loss_ray_np = torch.stack(loss_ray).cpu().detach().numpy()
    fig = plt.figure()
    plt.scatter( loss_ray_np[:,0], loss_ray_np[:,1] )
    plt.xlabel('Loss 1', fontsize=18)
    plt.ylabel('Loss 2', fontsize=18)
    save_fig_name = os.path.join(save_dir, 'loss_ray.pdf')
    plt.savefig( save_fig_name )
    print('save to {}'.format(save_fig_name))
    pickle_name = os.path.join(save_dir, 'res.pickle')
    with open(pickle_name, 'wb') as f:
            pickle.dump({
                'loss_history': loss_history,
                'loss_ray': loss_ray_np
            }, f)
    print('Pickle saved to {}'.format(pickle_name))













