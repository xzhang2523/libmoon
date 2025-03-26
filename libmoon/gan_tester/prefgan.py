import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
from torch.distributions.dirichlet import Dirichlet



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
    plt.plot([0,4], [0,4], linewidth=2, color='black')
    fig_name = os.path.join(folder_name, 'res_{:.2f}.pdf'.format(pref[0]))
    plt.savefig(fig_name, bbox_inches='tight')
    print('Save fig to {}'.format(fig_name))

# Generator: Transforms random noise into samples resembling the target distribution
class Generator(nn.Module):
    '''
        Pref-conditioned GAN
    '''
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        # 2 is the preference dimension.
        self.model = nn.Sequential(
            nn.Linear(input_dim+2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z, pref):
        batch_size = len(z)
        pref_batch = pref.repeat(batch_size, 1)
        input_arg = torch.cat([z, pref_batch], dim=1)
        # input_arg.shape: (64, 12)
        return self.model(input_arg)


# Discriminator: Classifies whether samples are real (from target Gaussian) or fake (from generator)
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Function to sample from a target Gaussian distribution
def sample_multiple_gaussian(batch_size, dim, mean=0, std=1):
    dist1 = torch.normal(mean=mean, std=std, size=(batch_size, dim))
    dist2 = torch.normal(mean=mean+4, std=std, size=(batch_size, dim))
    return dist1, dist2



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
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_obj = n_obj
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.generator = Generator(args.input_dim, args.output_dim)
        self.discriminator_arr = [Discriminator(args.output_dim) for _ in range(n_obj)]
        self.d_optimizer_arr = [optim.Adam(discriminator.parameters(), lr=args.lr) for discriminator in self.discriminator_arr]
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=args.lr)
        self.criterion = nn.BCELoss()

    def train(self):
        d_loss_arr = []
        for epoch in range(self.num_epochs):
            for _ in range(1):  # Training discriminator more than generator improves stability
                # Discriminator training
                real_samples_1, real_samples_2 = sample_multiple_gaussian(self.batch_size, self.output_dim)  # Real samples from Gaussian distribution
                real_samples_arr = [real_samples_1, real_samples_2]
                z = torch.randn(self.batch_size, self.input_dim)  # Random noise
                pref = Dirichlet(torch.Tensor([0.5, 0.5])).sample()
                fake_samples = self.generator(z, pref)  # Fake samples from generator
                for idx, discriminator in enumerate(self.discriminator_arr):
                    real_samples = real_samples_arr[idx]
                    d_real = discriminator(real_samples)
                    d_fake = discriminator(fake_samples.detach())  # Detach to avoid backpropagating through the generator
                    real_loss = self.criterion(d_real, torch.ones_like(d_real))
                    fake_loss = self.criterion(d_fake, torch.zeros_like(d_fake))
                    d_loss = (real_loss + fake_loss) / 2
                    self.d_optimizer_arr[idx].zero_grad()
                    d_loss.backward()
                    self.d_optimizer_arr[idx].step()
                # Generator training. Why here we need two z.
                z = torch.randn(self.batch_size, self.input_dim)
                fake_samples = self.generator(z, pref)
                g_loss_arr = []
                for idx, discriminator in enumerate(self.discriminator_arr):
                    d_fake = discriminator(fake_samples)
                    g_loss = self.criterion(d_fake, torch.ones_like(d_fake))
                    g_loss_arr.append(g_loss)
                g_loss_arr = torch.stack(g_loss_arr)
                self.g_optimizer.zero_grad()
                # g_loss_arr.shape: (2,1)
                # pref.shape: (2,1)
                scalar_loss = torch.dot(g_loss_arr, pref)
                scalar_loss.backward()
                self.g_optimizer.step()
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
                d_loss_arr.append(d_loss.item())

    def generate_samples(self, test_size):
        with torch.no_grad():
            z = torch.randn(test_size, self.input_dim)
            generated_samples = self.generator(z)
            real_samples_1, real_samples_2 = sample_multiple_gaussian(self.batch_size, self.output_dim)  # Real samples from Gaussian distribution
        return generated_samples.numpy(), real_samples_1, real_samples_2


    def evaluate(self, test_size, eval_num=11):
        '''
            :param eval_num, int: number of evaluation.
        '''
        pref_0 = np.linspace(0, 1, eval_num)
        # pref_0 = np.linspace(1, 0, eval_num)
        pref_1 = 1 - pref_0
        pref_arr = torch.Tensor(np.stack([pref_0, pref_1], axis=1))
        with torch.no_grad():
            '''
                For each preference, we need to generate samples.
            '''
            # print()
            for idx, pref in enumerate(pref_arr):
                z = torch.randn(test_size, self.input_dim)
                generated_samples = self.generator(z, pref)
                real_samples_1, real_samples_2 = sample_multiple_gaussian(self.batch_size, self.output_dim)
                fig = plt.figure()
                plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label='Generated')
                plt.scatter(real_samples_1[:, 0], real_samples_1[:, 1], label='Distribution 1')
                plt.scatter(real_samples_2[:, 0], real_samples_2[:, 1], label='Distribution 2')
                plt.xlabel('$X_1$', fontsize=20)
                plt.ylabel('$X_2$', fontsize=20)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                if idx==0:
                    plt.legend(fontsize=15)
                folder_name = 'D:\\pycharm_project\\libmoon\\Output\\divergence\\prefgan'
                os.makedirs(folder_name, exist_ok=True)
                fig_name = os.path.join(folder_name, 'res_{:.2f}.pdf'.format(pref[0]))
                plt.savefig(fig_name, bbox_inches='tight')
                print('Save fig to {}'.format(fig_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example script')
    parser.add_argument('--input-dim', type=int, default=10)     # What does it mean?
    parser.add_argument('--output-dim', type=int, default=2)
    parser.add_argument('--n-obj', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--pref0', type=float, default=0.2)
    # Hyperparameters
    args = parser.parse_args()
    # Model, optimizer, and loss function
    trainer = PrefGANTrainer(lr=args.lr, num_epochs=args.num_epochs, n_obj=args.n_obj,
                             batch_size=args.batch_size, input_dim=args.input_dim, output_dim=args.output_dim)
    trainer.train()
    # trainer.
    samples = trainer.evaluate(test_size=args.test_size, eval_num=11)
    folder_name = 'D:\\pycharm_project\\libmoon\\Output\\divergence'