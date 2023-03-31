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
    '''Add gaussian noise to the batch
    
    Input:
        batch: a batch of images (tensor)
        
    Output:
        noisy_batch: a batch of images with gaussian noise (tensor)'''
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
    '''Add saturation to the batch
    
    Input:
        batch: a batch of images (tensor)
        
    Output:
        saturated_batch: a batch of images with saturation (tensor)'''
    saturated_batch = torchvision.transforms.functional.adjust_saturation(batch, np.random.uniform(0,10))
    # for i in range(10):
    #     save_image(saturated_batch[i], "saturated_batch{}.png".format(i))
    return saturated_batch
def Gaussian_Blur(batch):
    '''Apply a random Gaussian Blur to the batch
    
    Input:
        batch: a batch of images (tensor)
        
    Output:
        blurred_batch: a batch of images with a random Gaussian Blur (tensor)'''
    blurred = GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    blurred_batch = blurred(batch)
    # for i in range(10):
    #     save_image(blurred_batch[i], "blurred_batch{}.png".format(i))
    return blurred_batch
def colorjiter(batch):
    '''Apply a random colorjitter to the batch
    
    Input:
        batch: a batch of images (tensor)
        
    Output:
        colorjiter_batch: a batch of images with a random colorjitter (tensor)'''
    b = np.random.uniform(0, 1); c = np.random.uniform(0, 1); s = np.random.uniform(0, 1); h = np.random.uniform(0, 1)
    colorjiter = ColorJitter(brightness=(max(0, 1-b),1+b), contrast=(max(0, 1-c),1+c), saturation=(max(0, 1-s),1+s), hue=(max(-0.5, 0-h),min(0.5,0+h)))
    colorjiter_batch = colorjiter(batch)
    # for i in range(10):
    #     save_image(colorjiter_batch[i], "colorjiter_batch{}.png".format(i))
    return colorjiter_batch
def invert(batch):
    '''Invert the batch
    
    Input:
        batch: a batch of images (tensor)
        
    Output:
        inverted_batch: a batch of inverted images (tensor)'''
    inversion = RandomInvert(1)
    inverted_batch = inversion(batch)
    # for i in range(10):
    #     save_image(inverted_batch[i], "inverted_batch{}.png".format(i))
    return inverted_batch
def augmentation(batch):
    '''Apply a random augmentation to the batch
    
    Input:
        batch: a batch of images (tensor)
        
    Output:
        augmented_batch: a batch of images with a random augmentation (tensor)'''
    funct = [gaussian_noise, saturation, Gaussian_Blur, colorjiter, invert]
    augmented_batch = choice(funct)(batch)
    return augmented_batch.type(torch.float32)
