# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import argparse
import numpy as np
from tqdm import tqdm

# 创建文件夹
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_img(x):
    # out = 0.5 * (x+0.5)
    img = make_grid(x, nrow=8, normalize=True).detach()
    # out = x.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    # out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return img

def numel(model):
    return sum(p.numel() for p in model.parameters())


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, args.z_dimension)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, args.z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(args.z_dimension, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))

        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder(out3)
        return out3, mean, logstd


def loss_function(recon_x, x, mean, std):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(std), 2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
    return BCE + KLD


if not os.path.exists('./img_VAE'):
    os.mkdir('./img_VAE')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    # parser.add_argument('--dataset-name', type=str, default='mnist')
    parser.add_argument('--data-name1', type=str, default='alarm')
    parser.add_argument('--data-name2', type=str, default='circle')

    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--z-dimension', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--pref0', type=float, default=0.0)
    args = parser.parse_args()
    # batch_size = 64
    # num_epoch = 15
    # z_dimension = 2
    # 图形啊处理过程
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.data_name1 == 'mnist':
        # mnist dataset mnist数据集下载
        mnist = datasets.MNIST(root='./data/', train=True, transform=img_transform, download=True)
        # data loader 数据载入
        dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=args.batch_size, shuffle=True)
    else:
        path1 = 'D:\\pycharm_project\\libmoon\\libmoon\\moogan\\data\\full_numpy_bitmap_{}.npy'.format(args.data_name1)
        img1_data = np.load(path1)
        img1_data = img1_data.reshape(-1, 1, 28, 28)
        img1_data = img1_data / 255

        path2 = 'D:\\pycharm_project\\libmoon\\libmoon\\moogan\\data\\full_numpy_bitmap_{}.npy'.format(args.data_name2)
        img2_data = np.load(path2)
        img2_data = img2_data.reshape(-1, 1, 28, 28)
        img2_data = img2_data / 255

        img1_data = torch.from_numpy(img1_data).to(torch.float).to(device)
        img2_data = torch.from_numpy(img2_data).to(torch.float).to(device)

        print('img1_data size: ', len(img1_data))
        dataloader = dataloader1 = torch.utils.data.DataLoader(img1_data, batch_size=args.batch_size, shuffle=True)
        dataloader2 = torch.utils.data.DataLoader(img2_data, batch_size=args.batch_size, shuffle=True)

    vae = VAE().to(device)

    num1 = numel(vae.encoder)
    num2 = numel(vae.decoder)
    print()

    # vae.load_state_dict(torch.load('./VAE_z2.pth'))
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    ###########################进入训练##判别器的判断过程#####################
    for epoch in range(args.n_epochs):  # 进行多个epoch的训练
        for i, (img, img2) in tqdm(enumerate(zip(dataloader, dataloader2))):
            num_img = img.size(0)
            # view()函数作用把img变成[batch_size,channel_size,784]
            img = img.view(num_img, 1, 28, 28).to(device)  # 将图片展开为28*28=784

            x, mean1, logstd1 = vae(img)  # 将真实图片放入判别器中
            loss = loss_function(x, img, mean1, logstd1)
            vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归 0.
            loss.backward()  # 将误差反向传播
            vae_optimizer.step()  # 更新参数

            num_img2 = img2.size(0)
            img2 = img2.view(num_img2, 1, 28, 28).to(device)  # 将图片展开为 28*28=784.
            x, mean2, logstd2 = vae(img2)  # 将真实图片放入判别器中
            loss = loss_function(x, img2, mean2, logstd2)
            vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
            loss.backward()  # 将误差反向传播
            vae_optimizer.step()  # 更新参数
            # mu1.shape: (64,2), logstd1.shape: (64,2)
            # try:
            if (i + 1) % 100 == 0:
                print('Epoch[{}/{}],vae_loss:{:.6f} '.format(
                    epoch, args.n_epochs, loss.item(),
                ))

            folder_name = os.path.join('D:\\pycharm_project\\libmoon\\libmoon\\moogan', 'img_VAE',
                                       '{}_{}'.format(args.data_name1, args.data_name2))
            os.makedirs(folder_name, exist_ok=True)

            if epoch == 0:
                real_images1 = make_grid(img[:25].cpu(), nrow=5, normalize=True).detach()
                save_image(real_images1, os.path.join(folder_name, 'real_images1.pdf'))

                real_images2 = make_grid(img2[:25].cpu(), nrow=5, normalize=True).detach()
                save_image(real_images2, os.path.join(folder_name, 'real_images2.pdf'))
            sample_size = 25
            pref0_arr = np.linspace(0, 1, 5)
            for pref0 in pref0_arr:
                meanA = torch.mean(mean1)
                meanB = torch.mean(mean2)
                mean = pref0 * meanA + (1 - pref0) * meanB
                stdA = torch.mean(torch.exp(logstd1))
                stdB = torch.mean(torch.exp(logstd2))
                std = pref0 * stdA + (1 - pref0) * stdB
                sample = torch.randn(sample_size, args.z_dimension).to(device) * std + mean
                output = vae.decoder_fc(sample)
                output = vae.decoder(output.view(output.shape[0], 32, 7, 7))
                fake_images = make_grid(output.cpu(), nrow=5, normalize=True).detach()
                fig_name = os.path.join(folder_name, 'fake_images_{}_{:.2f}.pdf'.format(epoch + 16, pref0))
                save_image(fake_images, fig_name)

    # 保存模型
    torch.save(vae.state_dict(), './VAE_z2.pth')
