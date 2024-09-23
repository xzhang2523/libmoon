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


def load_imgs():
    directory = '/mnt/d/pycharm/libmoon/libmoon/problem/mtl/mtl_data/FGNET/male/'
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

    def __init__(self, input_dim, output_dim, bs):
        super(Generator, self).__init__()
        # 2 is the preference dimension.
        self.f1 = nn.Linear(input_dim + 2, (input_dim + 2) * bs)
        self.FN1 = nn.Sequential(
            nn.Linear(input_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim))

    def forward(self, z, pref):
        batch_size = len(z)
        pref_batch = pref.repeat(batch_size, 1)
        input_arg = torch.cat([z, pref_batch], dim=1)
        input_arg = input_arg.view(batch_size, -1, input_arg.shape[1])
        res1 = self.f1(input_arg)
        res1 = res1.view(batch_size, -1, input_arg.shape[2])
        return self.FN1(res1)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512 // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512 // 2, 512 // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512 // 4, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


def sample_multiple_gaussian(batch_size, dim, mean=0, std=1):
    dist1 = torch.normal(mean=mean, std=std, size=(batch_size, dim))
    dist2 = torch.normal(mean=mean + 4, std=std, size=(batch_size, dim))
    return dist1, dist2


def sample_real_imgs(test_size):
    imgs = load_imgs()
    idx = np.random.choice(len(imgs), test_size)
    real_samples_1 = torch.Tensor(np.array([f[0] for f in imgs]))
    real_samples_2 = torch.Tensor(np.array([f[1] for f in imgs]))
    return real_samples_1[idx], real_samples_2[idx]


class PrefGANTrainer:
    def __init__(self, lr, num_epochs, batch_size, n_obj, input_dim, output_dim):
        '''
            :param lr, float: learning rate.
            :param num_epochs, int : number of epochs.
            :param batch_size, int : batch size.
            :param n_obj, int : number of objectives.
            :param pref, np.array : preference vector.
        '''
        self.lr = lr
        self.bs = int(32 * 32 / output_dim)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_obj = n_obj
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.generator = Generator(args.input_dim, args.output_dim, self.bs).to(device)
        self.discriminator_arr = [Discriminator(args.output_dim).to(device) for _ in range(n_obj)]
        self.d_optimizer_arr = [optim.Adam(discriminator.parameters(), lr=args.lr) for discriminator in
                                self.discriminator_arr]
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=args.lr)
        self.criterion = nn.BCELoss()

    def load_dataset(self):
        # real_samples_1, real_samples_2 = sample_multiple_gaussian(self.batch_size, self.output_dim)
        imgs = load_imgs()
        real_samples_1, real_samples_2 = (torch.Tensor(np.array([f[0] for f in imgs])),
                                          torch.Tensor(np.array([f[1] for f in imgs])))
        real_samples_arr = [real_samples_1, real_samples_2]
        return real_samples_arr

    def D_train(self, x, discriminator, opt):
        z = torch.randn(self.batch_size, self.input_dim).to(device)  # Random noise
        pref = Dirichlet(torch.Tensor([0.5, 0.5])).sample().to(device)
        discriminator.zero_grad()
        x_real = x.view(self.batch_size, -1, self.output_dim)
        d_real = discriminator(x_real)
        y_real = torch.ones_like(d_real)
        real_loss = self.criterion(d_real, y_real)
        x_fake = self.generator(z, pref)
        d_fake = discriminator(x_fake)
        y_fake = torch.zeros_like(d_fake)
        fake_loss = self.criterion(d_fake, y_fake)
        D_loss = real_loss + fake_loss
        D_loss.backward()
        opt.step()
        return D_loss

    def G_train(self, discriminator):
        z = torch.randn(self.batch_size, self.input_dim).to(device)  # Random noise
        pref = Dirichlet(torch.Tensor([0.5, 0.5])).sample().to(device)
        self.generator.zero_grad()
        fake_samples = self.generator(z, pref)
        d_fake = discriminator(fake_samples)
        y = torch.ones_like(d_fake)
        g_loss = self.criterion(d_fake, y)
        g_loss.backward()
        self.g_optimizer.step()
        return g_loss

    def train(self):
        real_samples_arr = self.load_dataset()
        for epoch in range(self.num_epochs):
            d_loss_arr = []
            for idx, discriminator in enumerate(self.discriminator_arr):
                real_samples = real_samples_arr[idx].to(device)
                d_loss = self.D_train(real_samples, discriminator, self.d_optimizer_arr[idx])
                d_loss_arr.append(d_loss.item())
            z = torch.randn(self.batch_size, self.input_dim).to(device)
            g_loss_arr = []
            for idx, discriminator in enumerate(self.discriminator_arr):
                g_loss = self.G_train(discriminator)
                g_loss_arr.append(g_loss)
            pref = Dirichlet(torch.Tensor([0.5, 0.5])).sample().to(device)
            g_loss_arr = torch.stack(g_loss_arr)
            # self.g_optimizer.zero_grad()
            # scalar_loss = torch.dot(g_loss_arr, pref)
            # scalar_loss.backward()
            # self.g_optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    def generate_samples(self, test_size):
        with torch.no_grad():
            z = torch.randn(test_size, self.input_dim)
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
                z = torch.randn(test_size, self.input_dim).to(device)
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
        with torch.no_grad():
            for idx, pref in enumerate(pref_arr):
                z = torch.randn(test_size, self.input_dim).to(device)
                generated_samples = self.generator(z, pref).cpu().numpy()
                real_samples_1, real_samples_2 = sample_real_imgs(test_size)
                folder_name = '/mnt/d/pycharm/libmoon/libmoon/Output/divergence/prefgan'
                os.makedirs(folder_name, exist_ok=True)
                # save generated samples
                for i in range(test_size):
                    img = real_samples_1[i]
                    fig = plt.figure()
                    plt.imshow(img, cmap='gray')
                    plt.axis('off')
                    fig_name = os.path.join(folder_name, 'res_real1_{:.2f}_{:.2f}.pdf'.format(i, pref[0]))
                    plt.savefig(fig_name, bbox_inches='tight')
                    print('Save fig to {}'.format(fig_name))
                for i in range(test_size):
                    img = real_samples_2[i]
                    fig = plt.figure()
                    plt.imshow(img, cmap='gray')
                    plt.axis('off')
                    fig_name = os.path.join(folder_name, 'res_real2_{:.2f}_{:.2f}.pdf'.format(i, pref[0]))
                    plt.savefig(fig_name, bbox_inches='tight')
                    print('Save fig to {}'.format(fig_name))
                for i in range(test_size):
                    img = generated_samples[i].reshape(32, 32)
                    fig = plt.figure()
                    plt.imshow(img, cmap='gray')
                    plt.axis('off')
                    fig_name = os.path.join(folder_name, 'res_gen_{:.2f}_{:.2f}.pdf'.format(i, pref[0]))
                    plt.savefig(fig_name, bbox_inches='tight')
                    print('Save fig to {}'.format(fig_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example script')
    parser.add_argument('--input-dim', type=int, default=10)  # What does it mean?
    parser.add_argument('--output-dim', type=int, default=128)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-size', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--pref0', type=float, default=0.2)
    # Hyperparameters
    args = parser.parse_args()
    trainer = PrefGANTrainer(lr=args.lr, num_epochs=args.num_epochs, n_obj=args.n_obj,
                             batch_size=args.batch_size, input_dim=args.input_dim, output_dim=args.output_dim)
    trainer.train()
    # samples = trainer.evaluate(test_size=args.test_size, eval_num=11)
    samples = trainer.evaluate_img(test_size=args.test_size, eval_num=11)
