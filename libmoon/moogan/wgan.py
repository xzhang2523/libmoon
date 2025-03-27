import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm


def numel(model):
    return sum(p.numel() for p in model.parameters())




os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n-epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--pref0", type=float, default=0.5, help="pref vectors")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
parser.add_argument("--data-name1", type=str, default='apple', help="interval betwen image samples")
parser.add_argument("--data-name2", type=str, default='baseball', help="interval betwen image samples")


opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

if __name__ == '__main__':
    # Initialize generator and discriminator
    generator = Generator()
    discriminator1, discriminator2 = Discriminator(), Discriminator()
    if cuda:
        generator.cuda()
        discriminator1.cuda()
        discriminator2.cuda()
    # Configure data loader
    # data_name = 'apple'
    # data_name = 'apple'
    if opt.data_name1 == 'mnist':
        os.makedirs("../../data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    else:
        path1 = 'D:\\pycharm_project\\libmoon\\libmoon\\moogan\\data\\full_numpy_bitmap_{}.npy'.format(opt.data_name1)

        img1_data = np.load(path1)
        img1_data = img1_data.reshape(-1, 1, 28, 28)
        img1_data = img1_data / 255 * 2 - 1

        print(img1_data)
        path2 = 'D:\\pycharm_project\\libmoon\\libmoon\\moogan\\data\\full_numpy_bitmap_{}.npy'.format(opt.data_name2)
        img2_data = np.load(path2)
        img2_data = img2_data.reshape(-1, 1, 28, 28)
        img2_data = img2_data / 255 * 2 - 1

        print('img1_data size: ', len(img1_data))
        dataloader1 = torch.utils.data.DataLoader(img1_data, batch_size=opt.batch_size, shuffle=True)
        dataloader2 = torch.utils.data.DataLoader(img2_data, batch_size=opt.batch_size, shuffle=True)


    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    optimizer_D1 = torch.optim.RMSprop(discriminator1.parameters(), lr=opt.lr)
    optimizer_D2 = torch.optim.RMSprop(discriminator2.parameters(), lr=opt.lr)

    num1,num2,mum3 = numel(generator), numel(discriminator1), numel(discriminator2)
    print('num of generator parameters: ', num1)



    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # ----------
    #  Training
    # ----------
    batches_done = 0
    for epoch in tqdm(range(opt.n_epochs)):
        # for i, imgs in enumerate(dataloader):
        for i, (imgs1, imgs2) in tqdm(enumerate(zip(dataloader1, dataloader2))):
            # Configure input
            # imgs.shape:
            # imgs.shape: (64, 1, 28, 28), range (-1,1)
            real_imgs1 = Variable(imgs1.type(Tensor))
            real_imgs2 = Variable(imgs2.type(Tensor))
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D1.zero_grad(), optimizer_D2.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs1.shape[0], opt.latent_dim))))
            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            # loss_D_arr = [0,0]
            # for obj_idx in range(2):
            loss_D1 = -torch.mean(discriminator1(real_imgs1)) + torch.mean(discriminator1(fake_imgs))
            loss_D2 = -torch.mean(discriminator2(real_imgs2)) + torch.mean(discriminator2(fake_imgs))

            loss_D = loss_D1 + loss_D2
            loss_D.backward()

            optimizer_D1.step()
            optimizer_D2.step()


            # Clip weights of discriminator
            for p in discriminator1.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            for p in discriminator2.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G1 = -torch.mean(discriminator1(gen_imgs))
                loss_G2 = -torch.mean(discriminator2(gen_imgs))
                loss_G = opt.pref0 * loss_G1 + (1-opt.pref0) * loss_G2
                loss_G.backward()
                optimizer_G.step()
                # print(
                #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                #     % (epoch, opt.n_epochs, batches_done % len(dataloader1), len(dataloader1), loss_D.item(), loss_G.item())
                # )

            if batches_done % opt.sample_interval == 0:
                print('epoch: {}, batches_done: {}'.format(epoch, batches_done))
                prefix_name = 'D:\\pycharm_project\\libmoon\\libmoon\\moogan'
                folder_name = os.path.join(prefix_name, 'images', '{}_{}'.format(opt.data_name1, opt.data_name2), 'pref_{:.2f}'.format(opt.pref0))
                os.makedirs(folder_name, exist_ok=True)
                img_name = os.path.join(folder_name, "{}.pdf".format(batches_done))
                save_image(-gen_imgs.data[:25], img_name, nrow=5, normalize=True)
                print(f"Saved {img_name}")
            batches_done += 1