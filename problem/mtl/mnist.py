import matplotlib.pyplot as plt

from util.constant import root_name
from problem.mtl.loaders.multimnist_loader import MultiMNISTData
import torch

from problem.mtl.objectives import CrossEntropyLoss
from problem.mtl.model.simple import MultiLeNet
loss_1 = CrossEntropyLoss(label_name='labels_l', logits_name='logits_l')
loss_2 = CrossEntropyLoss(label_name='labels_r', logits_name='logits_r')

from numpy import array
from tqdm import tqdm
import numpy as np


class MultiMnistProblem:

    # How to train at the same time.
    def __init__(self, args):
        self.dataset = MultiMNISTData('mnist', 'train')
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=10, shuffle=True, num_workers=0)
        self.dataset_test = MultiMNISTData('mnist', 'test')
        self.loader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=10, shuffle=True, num_workers=0)
        self.model = MultiLeNet([1, 36, 36])
        self.model.to(device)
        self.args = args
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def optimize(self):
        l1_history = []
        l2_history = []
        for epoch in tqdm(range(self.args.num_epoch)):

            l1_arr_np = []
            l2_arr_np = []

            for data in self.loader:
                data = { k: v.to(self.args.device) for k, v in data.items() }
                logits_dict = self.model( data )
                logits_dict['labels_l'] = data['labels_l']
                logits_dict['labels_r'] = data['labels_r']
                # logits_dict['logits_l'].shape: [batch_size, 10]

                l1 = loss_1( **logits_dict )
                l2 = loss_2( **logits_dict )

                self.optimizer.zero_grad()
                (l1+l2).backward()
                self.optimizer.step()

                l1_np = l1.cpu().detach().numpy()
                l2_np = l2.cpu().detach().numpy()

                l1_arr_np.append(l1_np)
                l2_arr_np.append(l2_np)

            l1_mean = np.mean(l1_arr_np,0)
            l2_mean = np.mean(l2_arr_np,0)

            l1_history.append(l1_mean)
            l2_history.append(l2_mean)
        return array(l1_history), array(l2_history)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--use-cuda', default=True, type=bool)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda")  # Use the GPU
        print('cuda is available')
    else:
        device = torch.device("cpu")  # Use the CPU
        print('cuda is not available')

    args.device = device
    problem = MultiMnistProblem(args)
    l1_history, l2_history = problem.optimize()
    plt.plot(l1_history, l2_history)
    plt.show()
    print('hello world')