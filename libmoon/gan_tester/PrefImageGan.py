import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import cv2
from torch.distributions.dirichlet import Dirichlet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


def load_FGNET_imgs():
    directory = '/home/a/yingyingyu/libmoon/libmoon/problem/mtl/mtl_data/FGNET/male/'
    Full_imgs = []
    males_f = os.listdir(directory)
    males_res = [[f[:3], f.split('.')[0][-2:]] for f in males_f if f.endswith('.JPG')]
    males = list(set([r[0] for r in males_res]))
    for m in males:
        imgs = [f for f in males_f if f.startswith(m)]
        imgs = sorted(imgs)
        try:
            img1 = cv2.imread(directory + imgs[0], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(directory + imgs[-1], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1, (32, 32))
            img2 = cv2.resize(img2, (32, 32))
            Full_imgs.append([img1, img2])
        except:
            continue
    females_f = os.listdir(directory.replace('male', 'female'))
    females_res = [[f[:3], f.split('.')[0][-2:]] for f in females_f if f.endswith('.JPG')]
    females = list(set([r[0] for r in females_res]))
    for f in females:
        imgs = [ff for ff in females_f if ff.startswith(f)]
        imgs = sorted(imgs)
        try:
            img1 = cv2.imread(directory + imgs[0], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(directory + imgs[-1], cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1, (32, 32))
            img2 = cv2.resize(img2, (32, 32))
            Full_imgs.append([img1, img2])
        except:
            continue
    return Full_imgs


def load_quickdraw_imgs():
    directory = '/home/a/yingyingyu/libmoon/libmoon/problem/mtl/mtl_data/quickdraw/'
    apple_f = np.load(directory + 'full_numpy_bitmap_apple.npy')
    baseball_f = np.load(directory + 'full_numpy_bitmap_baseball.npy')
    Full_imgs = []
    for i in range(100000):
        img1 = apple_f[i].reshape(28, 28)
        img2 = baseball_f[i].reshape(28, 28)
        Full_imgs.append([img1, img2])
    return Full_imgs


def plot_figure(folder_name, generated_samples, sample1, sample2, pref):
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label='Generated', s=50)
    plt.scatter(sample1[:, 0], sample1[:, 1], label='Sample 1', s=25, alpha=0.5)
    plt.scatter(sample2[:, 0], sample2[:, 1], label='Sample 2', s=25, alpha=0.5)
    if abs(pref[0]) < 1e-6:
        plt.legend(fontsize=20, loc='lower right')

    plt.xlabel('$X_1$', fontsize=25)
    plt.ylabel('$X_2$', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('equal')
    plt.plot([0, 4], [0, 4], linewidth=2, color='black')
    fig_name = os.path.join(folder_name, 'res_{:.2f}.pdf'.format(pref[0]))
    plt.savefig(fig_name, bbox_inches='tight')
    print('Save fig to {}'.format(fig_name))


# Generator: Transforms random noise into samples resembling the target distribution
class Generator(nn.Module):
    '''
        Pref-conditioned GAN
    '''

    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        def layer_block(input_size, output_size):
            layers = [nn.Linear(input_size, output_size)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.FN1 = nn.Sequential(
        #     nn.Linear(latent_dim + 2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dim))
        self.model = nn.Sequential(
            *layer_block(latent_dim + 2, 128),
            *layer_block(128, 256),
            *layer_block(256, 512),
            *layer_block(512, 1024),
            nn.Linear(1024, int(np.prod((1, 28, 28)))),
            nn.Tanh())

    def forward(self, z, pref):
        batch_size = len(z)
        pref_batch = pref.repeat(batch_size, 1)
        input_arg = torch.cat([z, pref_batch], dim=1)
        input_arg = input_arg.view(batch_size, -1, input_arg.shape[1])
        # res1 = self.f1(input_arg)
        img = self.model(input_arg)
        img = img.view(img.size(0), *(1, 28, 28))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod((1, 28, 28))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x):
        img_flat = x.view(x.size(0), -1)
        verdict = self.model(img_flat)
        return verdict


def sample_multiple_gaussian(batch_size, dim, mean=0, std=1):
    dist1 = torch.normal(mean=mean, std=std, size=(batch_size, dim))
    dist2 = torch.normal(mean=mean + 4, std=std, size=(batch_size, dim))
    return dist1, dist2


def sample_real_imgs(test_size):
    imgs = load_quickdraw_imgs()
    idx = np.random.choice(len(imgs), test_size)
    real_samples_1 = torch.Tensor(np.array([f[0] for f in imgs]))
    real_samples_2 = torch.Tensor(np.array([f[1] for f in imgs]))
    return real_samples_1[idx], real_samples_2[idx]


class PrefGANTrainer:
    def __init__(self, lr, num_epochs, batch_size, n_obj, latent_dim):
        '''
            :param lr, float: learning rate.
            :param num_epochs, int : number of epochs.
            :param batch_size, int : batch size.
            :param n_obj, int : number of objectives.
            :param pref, np.array : preference vector.
        '''
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_obj = n_obj
        self.latent_dim = latent_dim

        self.generator = Generator(self.latent_dim).to(device)
        self.discriminator_arr = [Discriminator().to(device) for _ in range(n_obj)]
        self.d_optimizer_arr = [optim.Adam(discriminator.parameters(), lr=self.lr) for discriminator in
                                self.discriminator_arr]
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()

    def load_dataset(self):
        imgs = load_quickdraw_imgs()
        real_samples_1, real_samples_2 = (torch.Tensor(np.array([f[0] for f in imgs])).view(-1, 28, 28),
                                          torch.Tensor(np.array([f[1] for f in imgs])).view(-1, 28, 28))
        real_samples_arr = [[real_samples_1[i], real_samples_2[i]] for i in range(len(real_samples_1))]
        dataset = ImageDataset(real_samples_arr)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return data_loader

    def D_train(self, x_real, discriminator, opt):
        z = torch.randn(self.batch_size, self.latent_dim).to(device)  # Random noise
        pref = Dirichlet(torch.Tensor([0.5, 0.5])).sample().to(device)
        discriminator.zero_grad()
        d_real = discriminator(x_real)
        y_real = torch.ones_like(d_real)
        real_loss = self.criterion(d_real, y_real)
        x_fake = self.generator(z, pref)
        d_fake = discriminator(x_fake)
        y_fake = torch.zeros_like(d_fake)
        fake_loss = self.criterion(d_fake, y_fake)
        D_loss = (real_loss + fake_loss) / 2
        D_loss.backward()
        opt.step()
        return D_loss

    def G_train(self, discriminator):
        z = torch.randn(self.batch_size, self.latent_dim).to(device)  # Random noise
        pref = Dirichlet(torch.Tensor([0.5, 0.5])).sample().to(device)
        self.generator.zero_grad()
        fake_samples = self.generator(z, pref)
        d_fake = discriminator(fake_samples)
        y = torch.ones_like(d_fake)
        g_loss = self.criterion(d_fake, y)
        # g_loss.backward()
        # self.g_optimizer.step()
        return g_loss

    def train(self):
        dataloader = self.load_dataset()
        for epoch in range(self.num_epochs):
            for i, images in enumerate(dataloader):
                d_loss_arr = []
                for idx, discriminator in enumerate(self.discriminator_arr):
                    real_samples = images[idx].to(device)
                    d_loss = self.D_train(real_samples, discriminator, self.d_optimizer_arr[idx])
                    d_loss_arr.append(d_loss.cpu().item())
                g_loss_arr = []
                g_loss_arr_ = []
                for idx, discriminator in enumerate(self.discriminator_arr):
                    g_loss = self.G_train(discriminator)
                    g_loss_arr.append(g_loss.cpu().item())
                    g_loss_arr_.append(g_loss)
                pref = Dirichlet(torch.Tensor([0.5, 0.5])).sample().to(device)
                g_loss_arr_ = torch.stack(g_loss_arr_)
                self.g_optimizer.zero_grad()
                scalar_loss = torch.dot(g_loss_arr_, pref)
                torch.autograd.set_detect_anomaly(True)
                scalar_loss.backward()
                self.g_optimizer.step()
                print(f"[Epoch {epoch:=4d}/{self.num_epochs}] [Batch {i:=4d}/{len(dataloader)}] ---> "
                      f"[D Loss: {np.mean(np.array(d_loss_arr)):.6f}] [G Loss: {np.mean(np.array(g_loss_arr)):.6f}]")

    def generate_samples(self, test_size):
        with torch.no_grad():
            z = torch.randn(test_size, self.latent_dim)
            generated_samples = self.generator(z)
            real_samples_1, real_samples_2 = sample_multiple_gaussian(self.batch_size, self.output_dim)
        return generated_samples.numpy(), real_samples_1, real_samples_2

    def evaluate(self, test_size, eval_num=11):
        '''
            :param eval_num, int: number of evaluation.
        '''
        pref_0 = np.linspace(0, 1, eval_num)
        pref_1 = 1 - pref_0
        pref_arr = torch.Tensor(np.stack([pref_0, pref_1], axis=1)).to(device)
        with torch.no_grad():
            '''
                For each preference, we need to generate samples.
            '''
            for idx, pref in enumerate(pref_arr):
                z = torch.randn(test_size, self.latent_dim).to(device)
                generated_samples = self.generator(z, pref).cpu().numpy()
                real_samples_1, real_samples_2 = sample_multiple_gaussian(self.batch_size, self.output_dim)
                fig = plt.figure()
                plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label='Generated')
                plt.scatter(real_samples_1[:, 0], real_samples_1[:, 1], label='Distribution 1')
                plt.scatter(real_samples_2[:, 0], real_samples_2[:, 1], label='Distribution 2')
                plt.xlabel('$X_1$', fontsize=20)
                plt.ylabel('$X_2$', fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                if idx == 0:
                    plt.legend(fontsize=15)
                folder_name = '/mnt/d/pycharm/libmoon/libmoon/Output/divergence/prefgan'
                os.makedirs(folder_name, exist_ok=True)
                fig_name = os.path.join(folder_name, 'res_{:.2f}.pdf'.format(pref[0]))
                plt.savefig(fig_name, bbox_inches='tight')
                print('Save fig to {}'.format(fig_name))

    def evaluate_img(self, test_size, eval_num=11):
        '''
            :param eval_num, int: number of evaluation.
        '''
        pref_0 = np.linspace(0, 1, eval_num)
        pref_1 = 1 - pref_0
        pref_arr = torch.Tensor(np.stack([pref_0, pref_1], axis=1)).to(device)
        folder_name = '/home/a/yingyingyu/libmoon/libmoon/Output/divergence/prefgan'
        os.makedirs(folder_name, exist_ok=True)
        real_samples_1, real_samples_2 = sample_real_imgs(test_size)
        # save generated samples
        for i in range(test_size):
            img = real_samples_1[i]
            fig = plt.figure()
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            fig_name = os.path.join(folder_name, 'res_real1_{:.2f}.pdf'.format(i))
            plt.savefig(fig_name, bbox_inches='tight')
            print('Save fig to {}'.format(fig_name))
        for i in range(test_size):
            img = real_samples_2[i]
            fig = plt.figure()
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            fig_name = os.path.join(folder_name, 'res_real2_{:.2f}.pdf'.format(i))
            plt.savefig(fig_name, bbox_inches='tight')
            print('Save fig to {}'.format(fig_name))
        with torch.no_grad():
            for idx, pref in enumerate(pref_arr):
                z = torch.randn(test_size, self.latent_dim).to(device)
                generated_samples = self.generator(z, pref).cpu().numpy()
                for i in range(test_size):
                    img = generated_samples[i].reshape(28, 28)
                    fig = plt.figure()
                    plt.imshow(img, cmap='gray')
                    plt.axis('off')
                    fig_name = os.path.join(folder_name, 'res_gen_{:.2f}_{:.2f}.pdf'.format(i, pref[0]))
                    plt.savefig(fig_name, bbox_inches='tight')
                    print('Save fig to {}'.format(fig_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example script')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-size', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--pref0', type=float, default=0.2)
    # Hyperparameters
    args = parser.parse_args()
    trainer = PrefGANTrainer(lr=args.lr, num_epochs=args.num_epochs, n_obj=args.n_obj,
                             batch_size=args.batch_size, latent_dim=args.latent_dim)
    trainer.train()
    # samples = trainer.evaluate(test_size=args.test_size, eval_num=11)
    samples = trainer.evaluate_img(test_size=args.test_size, eval_num=11)
