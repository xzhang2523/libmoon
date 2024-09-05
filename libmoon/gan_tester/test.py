import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

import os

def plot_figure(folder_name):
    plt.scatter(generated_samples_np[:, 0], generated_samples_np[:, 1], label='Generated samples')

    plt.scatter(sample1[:, 0], sample1[:, 1], label='Sample 1')
    plt.scatter(sample2[:, 0], sample2[:, 1], label='Sample 2')


    plt.legend(fontsize=18)
    plt.xlabel('$X_1$', fontsize=18)
    plt.ylabel('$X_2$', fontsize=18)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.axis('equal')

    fig_name = os.path.join(folder_name, 'res.pdf')
    plt.savefig(fig_name, bbox_inches='tight')
    print('Save fig to {}'.format(fig_name))
    plt.show()


# Generator: Transforms random noise into samples resembling the target distribution
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.model(z)

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



# GAN Training loop
def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, num_epochs, batch_size, input_dim, output_dim):
    d_loss_arr = []
    for epoch in range(num_epochs):
        # Discriminator training
        for _ in range(1):  # Training discriminator more than generator improves stability
            real_samples, _ = sample_multiple_gaussian(batch_size, output_dim)  # Real samples from Gaussian distribution
            z = torch.randn(batch_size, input_dim)  # Random noise
            fake_samples = generator(z)  # Fake samples from generator
            # Discriminator loss on real and fake samples
            d_real = discriminator(real_samples)
            d_fake = discriminator(fake_samples.detach())  # Detach to avoid backpropagating through the generator
            real_loss = criterion(d_real, torch.ones_like(d_real))
            fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
            d_loss = (real_loss + fake_loss) / 2
            # Backpropagation for discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Generator training
        z = torch.randn(batch_size, input_dim)
        fake_samples = generator(z)
        d_fake = discriminator(fake_samples)
        # Generator loss (want discriminator to classify fakes as real)
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        # Backpropagation for generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        # Logging
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
        d_loss_arr.append(d_loss.item())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example script')
    parser.add_argument('--input-dim', type=int, default=10)
    parser.add_argument('--output-dim', type=int, default=2)

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-size', type=int, default=100)

    parser.add_argument('--num-epochs', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-5)
    # Hyperparameters
    args = parser.parse_args()

    # Model, optimizer, and loss function
    generator = Generator(args.input_dim, args.output_dim)
    discriminator = Discriminator(args.output_dim)
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Train GAN
    train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, args.num_epochs,
              args.batch_size, args.input_dim, args.output_dim)

    sample1, sample2 = sample_multiple_gaussian(args.batch_size, args.output_dim)

    # Test generator
    z = torch.randn(args.test_size, args.input_dim)
    generated_samples = generator(z)

    # print("Generated samples: ", generated_samples)
    generated_samples_np = generated_samples.detach().numpy()

    folder_name = 'D:\\pycharm_project\\libmoon\\Output\\divergence'
    plot_figure(folder_name)
