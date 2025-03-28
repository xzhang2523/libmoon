import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import os
from torchvision.utils import make_grid
import argparse
import numpy as np
from tqdm import tqdm
from libmoon.util.constant import root_name
from modm_func import mokl
from libmoon.util.general import FolderDataset
from torch.utils.data import DataLoader

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_img(x):
    img = make_grid(x, nrow=8, normalize=True).detach()
    return img

def numel(model):
    return sum(p.numel() for p in model.parameters())


class VAE(nn.Module):
    def __init__(self, n_channels):
        self.n_channels = n_channels
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1),
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
            nn.ConvTranspose2d(16, self.n_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        # print('x.shape', x.shape)
        # assert False
        # x.shape: (64,3,28,28)
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        decoded_img = self.decoder_fc(z)
        decoded_img = decoded_img.view(decoded_img.shape[0], 32, 7, 7)
        decoded_img = self.decoder(decoded_img)
        return decoded_img.to(device), mean.to(device), logstd.to(device)


def vae_loss_function(recon_x, x, mean, std):
    _, n_channels, _, _ = x.size()
    BCE_arr = []
    for channel_idx in range(n_channels):
        # print('max recon_x', torch.max(recon_x[:, channel_idx, :, :]))
        # print('min recon_x', torch.min(recon_x[:, channel_idx, :, :]))
        # print('max x', torch.max(x[:, channel_idx, :, :]))
        # print('min x', torch.min(x[:, channel_idx, :, :]))
        BCE_i = F.binary_cross_entropy(recon_x[:, channel_idx, :, :], x[:, channel_idx, :, :],
                                                              reduction='sum')
        # print('BCE_i', BCE_i)
        BCE_arr.append( BCE_i )
    BCE = torch.sum(torch.stack(BCE_arr))
    var = torch.pow(torch.exp(std), 2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
    return BCE + KLD

if not os.path.exists('./img_VAE'):
    os.mkdir('./img_VAE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-name1', type=str, default='alarm')
    parser.add_argument('--data-name2', type=str, default='circle')
    parser.add_argument('--data-type', type=str, default='domainnet')     # Category: [domainnet, 'quickdraw']
    parser.add_argument('--domain-set-data', type=str, default='airplane')
    parser.add_argument('--domain1', type=str, default='real')
    parser.add_argument('--domain2', type=str, default='quickdraw')
    parser.add_argument('--n-epochs', type=int, default=5000)
    parser.add_argument('--z-dimension', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--pref0', type=float, default=0.0)
    args = parser.parse_args()
    # batch_size = 64, # num_epoch = 15, # z_dimension = 2
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if args.data_type == 'mnist':
        mnist = datasets.MNIST(root='./data/', train=True, transform=img_transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=args.batch_size, shuffle=True)
        n_channels = 1
    elif args.data_type == 'domainnet':
        path1 = os.path.join(
            root_name, 'libmoon', 'moogan', 'data', 'domainnet', args.domain1, args.domain_set_data
        )

        path2 = os.path.join(
            root_name, 'libmoon', 'moogan', 'data', 'domainnet', args.domain2, args.domain_set_data
        )

        dataset1 = FolderDataset(path1)
        dataset2 = FolderDataset(path2)
        print(dataset1[0].shape)
        print(dataset2[0].shape)
        print('len dataset1', len(dataset1))
        print('len dataset2', len(dataset2))

        dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
        n_channels = 3

    elif args.data_type == 'quickdraw':
        path1 = os.path.join(root_name, 'libmoon', 'moogan', 'data', 'quick_draw',
                             'full_numpy_bitmap_{}.npy'.format(args.data_name1))
        img1_data = np.load(path1)
        if args.data_type == 'quickdraw':
            img1_data = img1_data.reshape(-1, 1, 28, 28)

        img1_data = img1_data / 255
        path2 = os.path.join(root_name, 'libmoon', 'moogan', 'data', 'quick_draw',
                             'full_numpy_bitmap_{}.npy'.format(args.data_name2))
        img2_data = np.load(path2)
        img2_data = img2_data.reshape(-1, 1, 28, 28)
        img2_data = img2_data / 255
        img1_data = torch.from_numpy(img1_data).to(torch.float).to(device)
        img2_data = torch.from_numpy(img2_data).to(torch.float).to(device)
        print('img1_data size: ', len(img1_data))
        dataloader1 = DataLoader(img1_data, batch_size=args.batch_size, shuffle=True)
        dataloader2 = DataLoader(img2_data, batch_size=args.batch_size, shuffle=True)
        n_channels = 1
    else:
        assert False, 'dataset not implemented'

    vae = VAE(n_channels=n_channels).to(device)
    num1 = numel(vae.encoder)
    num2 = numel(vae.decoder)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for epoch in range(args.n_epochs):
        for i, (img1, img2) in tqdm(enumerate(zip(dataloader1, dataloader2))):
            img1 = img1.to(device)
            img2 = img2.to(device)
            num_img = img1.size(0)
            if args.data_type == 'quickdraw':
                img1 = img1.view(num_img, 1, 28, 28).to(device)  # 将图片展开为28*28=784

            decoded_img1, mean1, logstd1 = vae(img1)

            loss1 = vae_loss_function(decoded_img1, img1, mean1, logstd1)
            vae_optimizer.zero_grad()  # 在反向传播之前，先将梯度归 0.
            loss1.backward()
            vae_optimizer.step()  # 更新参数

            num_img2 = img2.size(0)
            if args.data_type == 'quickdraw':
                img2 = img2.view(num_img2, 1, 28, 28).to(device)  # 将图片展开为 28*28=784.

            decoded_img2, mean2, logstd2 = vae(img2)
            loss2 = vae_loss_function(decoded_img2, img2, mean2, logstd2)
            vae_optimizer.zero_grad()
            loss2.backward()
            vae_optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch[{}/{}],vae_loss:{:.6f} '.format(
                    epoch, args.n_epochs, loss.item(),
                ))

            if args.data_type == 'quickdraw':
                folder_name = os.path.join(root_name, 'libmoon', 'moogan', 'img_VAE', args.data_type,
                                           '{}_{}'.format(args.data_name1, args.data_name2))
            else:
                folder_name = os.path.join(root_name, 'libmoon', 'moogan', 'img_VAE', args.data_type, args.domain_set_data,
                                           '{}_{}'.format(args.domain1, args.domain2))

            os.makedirs(folder_name, exist_ok=True)
            if epoch == 0:
                real_images1 = make_grid(img1[:25].cpu(), nrow=5, normalize=True).detach()
                save_image(real_images1, os.path.join(folder_name, 'real_images1.pdf'))
                real_images2 = make_grid(img2[:25].cpu(), nrow=5, normalize=True).detach()
                save_image(real_images2, os.path.join(folder_name, 'real_images2.pdf'))
            # sample_size = 25
            pref0_arr = np.linspace(0, 1, 5)

            if i == 0:
                for pref0 in pref0_arr:
                    Std1 = torch.exp(logstd1)
                    Std2 = torch.exp(logstd2)
                    mu, std = mokl(mean1, mean2, Std1, Std2, pref0)
                    std = torch.diagonal(std, dim1=1, dim2=2)
                    sample_size = len(mu)
                    sample = torch.randn(sample_size, args.z_dimension).to(device) * std + mu
                    output = vae.decoder_fc(sample)
                    output = vae.decoder(output.view(output.shape[0], 32, 7, 7))
                    fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                    fig_name = os.path.join(folder_name, 'fake_images_{}_{:.2f}.pdf'.format(epoch + 16, pref0))
                    save_image(fake_images, fig_name)
                    print('img saved in', fig_name)
    # 保存模型
    torch.save(vae.state_dict(), './VAE_z2.pth')
