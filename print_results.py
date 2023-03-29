import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_augmentation import augmentation
from torchvision.utils import make_grid

#find txt files
def print_losses(name):
    '''Print the losses of the model and save the plot as a png file'''
    filelist = os.listdir()
    txtfiles = [file for file in filelist if file.endswith('.txt')]
    #read losses in txt files
    df = pd.read_csv(txtfiles[0], sep = ' ', header = None, names = [txtfiles[0][:-3]]).join(pd.read_csv(txtfiles[1], sep = ' ', header = None, names = [txtfiles[1][:-3]])).join(pd.read_csv(txtfiles[2], sep = ' ', header = None, names = [txtfiles[2][:-3]]))
    #plot losses
    plt.plot(df)
    plt.legend(df.columns)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses vs Epochs')
    plt.savefig(name + '.png')
def print_image_and_mask(image,label, model):
    pred = model(image)
    image_list = [image, augmentation(image), label, pred]  
    image_grid = make_grid(image_list, nrow=4)
if __name__ == '__main__':
    print_losses('losses')
