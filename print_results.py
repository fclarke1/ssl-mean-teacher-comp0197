import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_augmentation import augmentation
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

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
def print_image_and_mask(image,label, model=0):
    # pred = model(image)
    pred = torch.rand(1,256,256)
    image_list = [image, augmentation(image)]
    pred_list = [label, pred]
    # show(image) 
    # show(augmentation(image))
    # show(label)
    # show(pred)
    # image_grid = make_grid(image_list, nrow=2)
    # pred_grid = make_grid(pred_list, nrow=2)
    print_(image_list, ['Original', 'Augmented'], 'pictures.png')
    print_(pred_list, ['Label', 'Prediction'], 'mask.png')
def print_(images, labels, file_name):
    nb_image = len(images)
    images_size = images[0].shape[1]
    marge = 4
    text_box = 25
    image_comb = Image.new('RGB',(marge * (2 * nb_image) + nb_image * images_size , marge * 2 + images_size + text_box), color = 'white')
    draw = ImageDraw.Draw(image_comb)
    for i in range(nb_image):
        image= images[i].detach()
        image_comb.paste(F.to_pil_image(image), (marge + i * (images_size + 2 * marge), marge))
        draw.text((i * (images_size + marge * 2)+ marge, marge * 2 + images_size), labels[i], fill = 'black')
    image_comb.save(file_name)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
if __name__ == '__main__':
    print_losses('losses')
    image = torch.rand(3,256,256)
    label = torch.rand(1,256,256)
    print_image_and_mask(image,label)