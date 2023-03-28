import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import model_UNet
from Losses import Dice_loss, Unsup_Loss
from Utils import utils
from preprocessing_1_dataloader import get_data
from data_augmentation import augmentation, Gaussian_Blur
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable GPU support
def train(sup_per_cent, img_size, model_depth, batch_size, nb_epoch, ramp_up, consistancy, alpha, lr, lr_decay, wait_period, drop_out):
#### Hyper-Param ####
# data parms
    supervised_percent = sup_per_cent  # what percent of training is to be labelled
    img_resize = img_size             # resize all images to this size 
    is_only_labelled = False     # Training only on supervised data or not
    is_mixed_labelled = True
    # model params
    depth = model_depth       # depth of unet

    # Training params
    batch_size = batch_size
    epochs = nb_epoch
    ramp_up = ramp_up
    consistency = consistancy
    alpha = alpha
    global_step = 0
    lr = lr
    lr_gamma = lr_decay
    wait_period = wait_period

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
        losses.append([epoch, running_loss, running_loss_sup, running_loss_unsup])
        if epoch % eval_freq == 0:
            accuracy, IOU = utils.eval_model(modelS, val_loader, device)
            accs.append(accuracy)
            IOUs.append(IOU)
            print(f'accuracy: {accuracy:2.0%}, IOU: {IOU:2.0%}')
    table = {'Epoch' : losses[:,0], 'running loss': losses[:,1], 'running loss sup': losses[:,2], 'running loss unsup': losses[:,3], 'accuracy':accs,'IOU':IOU}
    df = pd.DataFrame(table)
    filename = f'supervized_{sup_per_cent:.2%}_img_size_{img_size}_model_depth_{model_depth}_batch_size_{batch_size}_nb_epoch_{nb_epoch}_ramp_up_{ramp_up}_consistancy_{consistancy}_alpha_{alpha}_lr_{lr}_lr_decay_{lr_decay}_wait_period_{wait_period}_drop_out_{drop_out}.csv'
    df.to_csv(filename + '.csv')
    torch.save(modelS.state_dict(), filename + '.pt')
if __name__ == "__main__":
    train(0.1, 128, 4, 8, 100, 10, 10, 0.999, 0.001, 0.9, 10, 0.5)                   

