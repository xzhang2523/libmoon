import torch
import pickle
from sklearn.model_selection import train_test_split
# taken from https://github.com/Xi-L/ParetoMTL and adapted
import os

from libmoon.util_global.constant import root_name


class MultiMNISTData(torch.utils.data.Dataset):
    """
    The datasets from ParetoMTL
    """
    def __init__(self, dataset, split, root='data/multi', **kwargs):
        assert dataset in ['mnist', 'fashion', 'fmnist']
        assert split in ['train', 'val', 'test']

        # equal size of val and test split
        train_split = .9

        if dataset == 'mnist':
            self.path = os.path.join(root_name, 'problem', 'mtl', 'data', 'multimnist', 'mnist.pickle')

        elif dataset == 'fashion':
            self.path = os.path.join(root_name, 'problem', 'mtl', 'data', 'multimnist', 'fashion.pickle')

        elif dataset == 'fmnist':
            self.path = os.path.join(root_name, 'problem', 'mtl', 'data', 'multimnist', 'fmnist.pickle')


        self.val_size = .1
        with open(self.path, 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

        n_train = len(trainX)
        if self.val_size > 0:
            trainX, valX, trainLabel, valLabel = train_test_split(
                trainX, trainLabel, test_size=self.val_size, random_state=42
            )
            n_train = len(trainX)
            n_val = len(valX)

        trainX = torch.from_numpy(trainX.reshape(n_train, 1, 36, 36)).float()
        trainLabel = torch.from_numpy(trainLabel).long()
        testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
        testLabel = torch.from_numpy(testLabel).long()

        if self.val_size > 0:
            valX = torch.from_numpy(valX.reshape(n_val, 1, 36, 36)).float()
            valLabel = torch.from_numpy(valLabel).long()

        if split in ['train', 'val']:
            n = int(len(trainX) * train_split)
            if split == 'val':
                self.X = valX
                self.y = valLabel
            elif split == 'train':
                self.X = trainX
                self.y = trainLabel
        elif split == 'test':
            self.X = testX
            self.y = testLabel

    def __getitem__(self, index):
        return dict(data=self.X[index], labels_l=self.y[index, 0], labels_r=self.y[index, 1])

    def __len__(self):
        return len(self.X)

    def task_names(self):
        return ['l', 'r']







if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dst = MultiMNISTData(dataset='mnist', split='val')
    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=0)

    for dat in loader:
        ims = dat['data'].view(10, 36, 36).numpy()
        labs_l = dat['labels_l']
        labs_r = dat['labels_r']

        f, axarr = plt.subplots(2, 5)
        for j in range(5):
            for i in range(2):
                axarr[i][j].imshow(ims[j * 2 + i, :, :], cmap='gray')
                axarr[i][j].set_title('{}_{}'.format(labs_l[j * 2 + i], labs_r[j * 2 + i]))
        plt.show()
        a = input()

        if a == 'ex':
            break
        else:
            plt.close()

