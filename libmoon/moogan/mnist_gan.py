import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def display_images(images, n_cols=4, figsize=(12, 6)):
    """
        Utility function to display a collection of images in a grid
        Parameters
        ----------
        images: Tensor
                tensor of shape (batch_size, channel, height, width)
                containing images to be displayed
        n_cols: int
                number of columns in the grid
        Returns
        -------
        None
    """
    plt.style.use('ggplot')
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)
    plt.figure(figsize=figsize)
    for idx in range(n_images):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        image = images[idx]
        # make dims H x W x C
        image = image.permute(1, 2, 0)
        cmap = 'gray' if image.shape[2] == 1 else plt.cm.viridis
        ax.imshow(image, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='./data',
                              train=True,
                              download=True,
                              transform=transform)

    print(train_ds.data.shape)
    print(train_ds.targets.shape)
    print(train_ds.classes)
    print(train_ds.data[0])
    print(train_ds.targets[0])
    print(train_ds.data[0].max())
    print(train_ds.data[0].min())
    print(train_ds.data[0].float().mean())
    print(train_ds.data[0].float().std())

    # Build dataloader
    dl = DataLoader(dataset=train_ds, shuffle=True, batch_size=64)

    image_batch = next(iter(dl))
    print(len(image_batch), type(image_batch))
    print(image_batch[0].shape)
    print(image_batch[1].shape)

    # display_images(images=image_batch[0], n_cols=8)













