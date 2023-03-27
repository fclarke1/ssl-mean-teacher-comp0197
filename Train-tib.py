import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import model_UNet
from Losses import Dice_loss, Unsup_Loss
from Utils import utils
from preprocessing_1_dataloader import get_data
from data_augmentation import augmentation, Gaussian_Blur
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable GPU support

#### Hyper-Param ####
# data parms
supervised_percent = 0.2  # what percent of training is to be labelled
img_resize = 32             # resize all images to this size 
is_only_labelled = False     # Training only on supervised data or not
is_mixed_labelled = True
# model params
depth = 3       # depth of unet

# Training params
batch_size = 2
epochs = 100
ramp_up = 5
consistency = 56
alpha = 0.999
global_step = 0
lr = 1e-3
lr_gamma = 0.9
wait_period = 5

#create 2 network
modelS = model_UNet.UNet(in_channels=3, num_classes=2, depth=depth)
modelS = modelS.to(device)
modelT = model_UNet.UNet(in_channels=3, num_classes=2, depth=depth)
modelT = modelT.to(device)
#create the losses
sup_crit = nn.CrossEntropyLoss().to(device)
unsup_crit = nn.CrossEntropyLoss().to(device)
#optimizer
optimizer = Adam(modelS.parameters())
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma, last_epoch=-1, verbose=True)
##data loader
mixed_train_loader, val_loader, test_loader = get_data(supervised_percent,1-supervised_percent,0.2,0.1, batch_size = batch_size, img_resize = img_resize, is_mixed_loader = is_mixed_labelled)

#Train
eval_freq = 1
losses, accs, IOUs = [], [], []
for epoch in range(epochs):
    modelS.train()
    running_loss = 0
    running_loss_sup = 0
    running_loss_unsup = 0
    w_t = utils.wt(rampup_length = ramp_up, current = epoch, alpha = alpha, wait_period = wait_period)
    for step, data in enumerate(mixed_train_loader):
        imgs, labs = data
        #Augmentation
        imgS_aug= augmentation(imgs)
        imgS_aug = imgS_aug.to(device)
        labs = labs.squeeze().type(torch.LongTensor).to(device)
        
        optimizer.zero_grad()
    
        #Forward pass for student and teacher
        z = modelS(imgS_aug)

        #Finding images with label
        sup_idx = np.array(torch.tensor([elem != -1 for elem in labs[:, 0, 0, 0]])) #If batchsize is the first dim

        if sup_idx.sum() == 0:
            print('sup_idx.sum() == 0')
            Ls = torch.tensor(0)
        else:
            # Ls = sup_crit(z[sup_idx], labs[sup_idx])
            Ls = Dice_loss.dice_loss(z[sup_idx], labs[sup_idx])
        Lu = Unsup_Loss.unsup_loss(z, modelT, imgs, unsup_crit, device)

        loss = Ls + w_t * Lu
        loss.backward()

        optimizer.step()

        global_step += 1
        utils.update_ema_variables(modelS, modelT, alpha, global_step)
        running_loss += loss.item()
        running_loss_sup += Ls.item()
        running_loss_unsup += Lu.item()
        scheduler.step()

    print(f'Epoch {epoch + 1:4d} - Loss: {running_loss:6.2f}, loss_up: {running_loss_sup:6.2f}, loss_unsup: {running_loss_unsup:6.2f}')
    losses.append(running_loss)
    if epoch % eval_freq == 0:
        accuracy, IOU = utils.eval_model(modelS, val_loader, device)
        accs.append(accuracy)
        IOUs.append(IOU)
        print(f'accuracy: {accuracy:2.0%}, IOU: {IOU:2.0%}')
                

