'''
https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
'''
import numpy as np
from torchvision.utils import save_image
import torch
import torchvision
from random import choice
def gaussian_noise(batch):
    noise = np.random.normal(0, np.random.uniform(0.02,0.1), batch.shape)
    noisy_batch = batch + noise
    noisy_batch = np.clip(noisy_batch, 0, 1)
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
    blurred = torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    blurred_batch = blurred(batch)
    # for i in range(10):
    #     save_image(blurred_batch[i], "blurred_batch{}.png".format(i))
    return blurred_batch
def augmentation(batch):
    funct = [gaussian_noise, saturation, Gaussian_Blur]
    augmented_batch = choice(funct)(batch)
    return augmented_batch