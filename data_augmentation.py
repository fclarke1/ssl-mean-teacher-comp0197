'''
https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
'''
import numpy as np
from torchvision.utils import save_image
import torch
import torchvision
from torchvision.transforms import ColorJitter, GaussianBlur, RandomInvert
from random import choice
def gaussian_noise(batch):
    min_batch = batch.min()
    max_batch = batch.max()
    noise = np.random.normal(0, np.random.uniform(0.02,0.1), batch.shape)
    noisy_batch = batch + noise
    noisy_batch = np.clip(noisy_batch, min_batch, max_batch)
    # for i in range(10):
    #     save_image(batch[i], "batch_{}.png".format(i))
    # #     save_image(torch.from_numpy(noise[i]), "noisy_{}.png".format(i))
    #     save_image(noisy_batch[i], "noisy_batch_{}.png".format(i))
    return noisy_batch
def saturation(batch):
    saturated_batch = torchvision.transforms.functional.adjust_saturation(batch, np.random.uniform(0,10))
    # for i in range(10):
    #     save_image(saturated_batch[i], "saturated_batch{}.png".format(i))
    return saturated_batch
def Gaussian_Blur(batch):
    blurred = GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 5))
    blurred_batch = blurred(batch)
    # for i in range(10):
    #     save_image(blurred_batch[i], "blurred_batch{}.png".format(10+i))
    return blurred_batch
def colorjiter(batch):
    b = np.random.uniform(0, 1); c = np.random.uniform(0, 1); s = np.random.uniform(0, 1); h = np.random.uniform(0, 1)
    colorjiter = ColorJitter(brightness=(max(0, 1-b),1+b), contrast=(max(0, 1-c),1+c), saturation=(max(0, 1-s),1+s), hue=(max(-0.5, 0-h),min(0.5,0+h)))
    colorjiter_batch = colorjiter(batch)
    # for i in range(10):
    #     save_image(colorjiter_batch[i], "colorjiter_batch{}.png".format(i))
    return colorjiter_batch
def invert(batch):
    inversion = RandomInvert(1)
    inverted_batch = inversion(batch)
    # for i in range(10):
    #     save_image(inverted_batch[i], "inverted_batch{}.png".format(i))
    return inverted_batch
def augmentation(batch):
    funct = [gaussian_noise, saturation, Gaussian_Blur, colorjiter, invert]
    augmented_batch = choice(funct)(batch)
    return augmented_batch.type(torch.float32)
if __name__ == "__main__":
    from preprocessing_2_dataloaders import get_data
    labeled_train_loader, unlabeled_train_loader, val_loader, test_loader = get_data(0.2, 0.8, 0.2, 0.2)