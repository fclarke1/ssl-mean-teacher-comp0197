import os
import re
import torch
import pandas as pd
import numpy as np
import model_UNet
import matplotlib.pyplot as plt
from data_augmentation import augmentation
from data_into_loaders import get_data
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

def print_losses(name):
    '''Print the losses of the model and save the plot as a png file'''
    filelist = os.listdir()
    txtfiles = [file for file in filelist if file.endswith('loss')]
    if len(txtfiles) == 0:
        return
    #read losses in txt files
    df = join_txt(txtfiles)
    df = df.set_index(pd.Index(range(1, len(df)+1)))
    #plot and save dataframe
    save_plot(df, name, 'Epoch', 'Loss', 'Losses vs Epochs')
def print_metrics(name):
    '''Print the metrics of the model and save the plot as a png file'''
    filelist = os.listdir()
    txtfiles = [file for file in filelist if (file.endswith('IOU') or file.endswith('acc') or file.endswith('accuracy'))]
    if len(txtfiles) == 0:
        return
    #read metrics in txt files
    df = join_txt(txtfiles)
    df = df.set_index(pd.Index(list(5 * np.arange(0, len(df)) + 1)))
    #plot and save dataframe
    save_plot(df, name, 'Epoch', 'Accuracy', 'Accuracy and IOU vs Epochs', dash = True)
def print_weights(name):
    '''Print the weights of the model and save the plot as a png file'''
    filelist = os.listdir()
    txtfiles = [file for file in filelist if file.endswith('weights')]
    if len(txtfiles) == 0:
        return
    #read weights in txt files
    df = join_txt(txtfiles)
    df = df.set_index(pd.Index(range(1, len(df)+1)))
    #plot and save dataframe
    save_plot(df, name, 'Epoch', 'Weights', 'Weights vs Epochs')
def save_plot(df, name, x_label, y_label, title, dash = False):
    '''Save the plot of the dataframe as a png file'''
    for i in range(len(df.columns)):
        if dash and i >= len(df.columns)//2:
            plt.plot(df.iloc[:,i], linestyle = '--')
        else:
            plt.plot(df.iloc[:,i])
    plt.legend(df.columns, loc = 1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(name + '.png')
    plt.clf()
def join_txt(txtfiles):
    '''Join the txt files in a dataframe'''
    df = pd.read_csv(txtfiles[0], sep = ' ', header = None, names = [txtfiles[0]])
    for i in range(1, len(txtfiles)):
        df = df.join(pd.read_csv(txtfiles[i], sep = ' ', header = None, names = [txtfiles[i]]))
    return df
def print_image_and_mask(image,label, model=0, depth = 3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = model_UNet.UNet(in_channels = 3, num_classes = 2, depth = depth)
    network.load_state_dict(torch.load(model, map_location = device))
    with torch.no_grad():
        pred = torch.argmax(network(image.unsqueeze(0)),1).float()
    image_0_1 = RGB_0_1(image)
    image_list = [image_0_1, augmentation(image_0_1)]
    pred_list = [label.unsqueeze(1), pred]
    print_(image_list, ['Original', 'Augmented'], 'pictures.png')
    print_(pred_list, ['Label', 'Prediction'], 'mask.png')
def print_(images, labels, file_name):
    nb_image = len(images)
    images_size = images[0].shape[-1]
    marge = 4
    text_box = 25
    image_comb = Image.new('RGB',(marge * (2 * nb_image) + nb_image * images_size , marge * 2 + images_size + text_box), color = 'white')
    draw = ImageDraw.Draw(image_comb)
    for i in range(nb_image):
        image= images[i].detach().squeeze(0)
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
def RGB_0_1(images):
    '''Convert images from [0,1] to [0,255]'''
    images = images.clone()
    value_range = images.min(), images.max()
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))
    def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))
    if len(images.shape) == 4:
        for t in images:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(images, value_range)
    return images
if __name__ == '__main__':
    supervised_pct = 0.25
    val_pct, test_pct = 0.2, 0.1
    batch_size = 32
    img_resize = 64
    mixed_train_loader, val_loader, test_loader = get_data(supervised_pct,1 - supervised_pct, val_pct, test_pct, batch_size=batch_size, img_resize=img_resize)
    itertest = iter(test_loader)
    images, labels = next(itertest)
    # images = torch.rand(32,3,64,64)
    # labels = torch.rand(32,1,64,64)
    image = images[0]
    label = labels[0]
    # image = torch.rand(3,64,64)
    # label = torch.rand(1,64,64)
    files_list=os.listdir()
    for file in files_list:
        if file.find('.') == -1 and file != 'data' and file != '__pycache__':
            os.chdir(file)
            models_list = os.listdir()
            for model in models_list:
                os.chdir(model)
                print_weights('weights')
                print_metrics('Accuracy and IOU')
                print_losses('losses')
                list_of_files = os.listdir()
                weights_files = []
                for file in list_of_files:
                    if file.endswith('.pt'):
                        weights_files.append(file)
                if weights_files != []:
                    weights_files = sorted(weights_files, key=lambda s: int(re.search(r'\d+', s).group()))
                print_image_and_mask(image,label, weights_files[-1], depth = 3)
                os.chdir('..')
            os.chdir('..')