import cv2
import matplotlib.pyplot as plt


def plot_figure(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def load_imgs_path():
    import os
    directory = '/mnt/d/pycharm/libmoon/libmoon/problem/mtl/mtl_data/FGNET/male/'
    males_f = os.listdir(directory)
    males_res = [[f[:3], f.split('.')[0][-2:]] for f in males_f if f.endswith('.JPG')]
    males = list(set([r[0] for r in males_res]))
    males_imgs = []
    for m in males:
        imgs = [f for f in males_f if f.startswith(m)]
        imgs = sorted(imgs)
        males_imgs.append([directory + imgs[0], directory + imgs[-1]])
    females_f = os.listdir(directory.replace('male', 'female'))
    females_res = [[f[:3], f.split('.')[0][-2:]] for f in females_f if f.endswith('.JPG')]
    females = list(set([r[0] for r in females_res]))
    females_imgs = []
    for f in females:
        imgs = [ff for ff in females_f if ff.startswith(f)]
        imgs = sorted(imgs)
        females_imgs.append([directory + imgs[0], directory + imgs[-1]])
    males_imgs.extend(females_imgs)
    return males_imgs


if __name__ == '__main__':
    imgs = load_imgs_path()
    plot_figure(imgs[0][0])
    plot_figure(imgs[0][1])
