import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
class ImgGenerator(nn.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        image_channels = 1
        super(ImgGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.model(z).reshape(-1, 28, 28).to(torch.float) * 255

# Discriminator: Classifies whether samples are real (from target Gaussian) or fake (from generator)
class ImgDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(ImgDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Add sequence of convolutional and fully connected layers
        x = x.reshape(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


# Function to sample from a target Gaussian distribution
# def sample_multiple_imgs(batch_size):
#     # dist1 = torch.normal(mean=mean, std=std, size=(batch_size, dim))
#     # dist2 = torch.normal(mean=mean + 4, std=std, size=(batch_size, dim))
#     path_1 = 'D:\\pycharm_project\\libmoon\\libmoon\\moogan\\data\\full_numpy_bitmap_apple.npy'
#     path_2 = 'D:\\pycharm_project\\libmoon\\libmoon\\moogan\\data\\full_numpy_bitmap_baseball.npy'
#
#     img1_data = np.load(path_1)
#     img2_data = np.load(path_2)
#     data_loader1 = torch.utils.data.DataLoader(img1_data, batch_size=batch_size, shuffle=True)
#     data_loader2 = torch.utils.data.DataLoader(img2_data, batch_size=batch_size, shuffle=True)
#     # res1 = iter(data_loader1)
#     # res2 = iter(data_loader2)
#     # res1 = next(data_loader1)
#     # res2 = next(data_loader2)
#     for res1, res2 in zip(data_loader1, data_loader2):
#         return res1.to(torch.float), res2.to(torch.float)

class MOGANImgTrainer:
    def __init__(self, lr_d, lr_g, num_epochs, batch_size, n_obj, pref, input_dim):
        '''
            :param lr, float: learning rate.
            :param num_epochs, int : number of epochs.
            :param batch_size, int : batch size.
            :param n_obj, int : number of objectives.
            :param pref, np.array : preference vector.
        '''
        # self.lr = lr
        self.lr_d = lr_d
        self.lr_g = lr_g

        self.pref = pref
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_obj = n_obj
        self.input_dim = input_dim

        self.generator = ImgGenerator(args.input_dim)
        self.discriminator_arr = [ImgDiscriminator(input_dim=args.input_dim) for _ in range(n_obj)]
        self.d_optimizer_arr = [optim.Adam(discriminator.parameters(), lr=args.lr_d) for discriminator in self.discriminator_arr]                             # Discriminator optimizer
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=args.lr_g)    # Generator optimizer
        self.criterion = nn.BCELoss(reduction='mean')

        path_1 = '/libmoon/moogan/data/full_numpy_bitmap_apple.npy'
        path_2 = '/libmoon/moogan/data/full_numpy_bitmap_baseball.npy'
        img1_data = np.load(path_1)
        img1_data = img1_data.reshape(-1, 28, 28)
        img2_data = np.load(path_2)
        img2_data = img2_data.reshape(-1, 28, 28)
        print('img1_data size: ', len(img1_data))
        self.data_loader1 = torch.utils.data.DataLoader(img1_data, batch_size=batch_size, shuffle=True)
        self.data_loader2 = torch.utils.data.DataLoader(img2_data, batch_size=batch_size, shuffle=True)

    def train(self):
        d_loss_arr = []
        for epoch in range(self.num_epochs):
            for batch_idx, (real_samples_1, real_samples_2) in tqdm(enumerate(zip(self.data_loader1, self.data_loader2)) ):
                real_samples_1 = real_samples_1.to(torch.float)
                real_samples_2 = real_samples_2.to(torch.float)
                real_samples_arr = [real_samples_1, real_samples_2]
                z = torch.randn(self.batch_size, self.input_dim)  # Random noise
                fake_samples = self.generator(z)  # Fake samples from generator
                for idx, discriminator in enumerate(self.discriminator_arr):
                    real_samples = real_samples_arr[idx]
                    # real_samples.shape: (64, 28, 28)
                    d_real = discriminator(real_samples)
                    d_fake = discriminator( fake_samples.detach() )  # Detach to avoid backpropagating through the generator
                    real_loss = self.criterion(d_real, torch.ones_like(d_real))
                    fake_loss = self.criterion(d_fake, torch.zeros_like(d_fake))
                    # print('real_loss: ', real_loss.detach().numpy() )
                    # print('fake_loss: ', fake_loss.detach().numpy() )
                    d_loss = (real_loss + fake_loss) / 2
                    # print('d_loss: ', d_loss.detach().numpy())
                    self.d_optimizer_arr[idx].zero_grad()
                    d_loss.backward()
                    self.d_optimizer_arr[idx].step()
                # Generator training
                z = torch.randn(self.batch_size, self.input_dim)
                fake_samples = self.generator(z)
                g_loss_arr = []
                for idx, discriminator in enumerate(self.discriminator_arr):
                    d_fake = discriminator(fake_samples)
                    g_loss = self.criterion(d_fake, torch.ones_like(d_fake))
                    g_loss_arr.append(g_loss)

                g_loss_arr = torch.stack(g_loss_arr)
                self.g_optimizer.zero_grad()
                scalar_loss = torch.dot(g_loss_arr, torch.Tensor(self.pref))
                scalar_loss.backward()
                self.g_optimizer.step()

                # Logging.
                if (batch_idx + 1) % 50 == 0:
                    print(
                        f'Epoch [{batch_idx + 1}/{self.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
                d_loss_arr.append(d_loss.item())

    def generate_samples(self, test_size):
        with torch.no_grad():
            z = torch.randn(test_size, self.input_dim)
            generated_samples = self.generator(z)
            # real_samples_1, real_samples_2 = sample_multiple_imgs(self.batch_size,
            #                                                           self.output_dim)  # Real samples from Gaussian distribution
        return generated_samples.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example script')
    parser.add_argument('--input-dim', type=int, default=25)  # What does it mean?
    # parser.add_argument('--output-dim', type=int, default=2)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=2000)

    parser.add_argument('--lr-d', type=float, default=1e-6)
    parser.add_argument('--lr-g', type=float, default=1e-4)
    parser.add_argument('--pref0', type=float, default=1.0)

    # Hyperparameters
    args = parser.parse_args()
    pref = np.array([args.pref0, 1-args.pref0])
    print('Preference: ', pref)
    # Model, optimizer, and loss function
    trainer = MOGANImgTrainer(lr_d=args.lr_d, lr_g=args.lr_g, num_epochs=args.num_epochs, n_obj=args.n_obj, pref=pref,
                              batch_size=args.batch_size, input_dim=args.input_dim,)

    trainer.train()
    generate_samples = int(trainer.generate_samples(args.test_size))
    fig, axes = plt.subplots(nrows=2, ncols=8)
    for i in range(16):
        ax = axes[i // 8, i % 8]
        ax.imshow(generate_samples[i])
        ax.axis('off')
    plt.show()

    # folder_name = 'D:\\pycharm_project\\libmoon\\Output\\divergence'
    # os.makedirs(folder_name, exist_ok=True)
    # plot_figure(folder_name, generate_samples, sample1, sample2, pref)