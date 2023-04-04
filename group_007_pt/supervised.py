import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import model_UNet
from data_augmentation import augmentation
from data_into_loaders import get_data
from utils import dice_loss, wt, update_ema_variables, unsup_loss, evaluate_model



def supervised_train(pct_data):
    """
    Given the percentage of training dataset being used and labelled (the remaining train dataset will be not used)
    this function trains a UNet model using Supervised learning with the dice loss
    The model trained is saved in the folder "/models/model_name"
    
    supervised_pct (float): percentage of the training dataset being used to train (all labelled) in (0,1]
    """
    
    pct_data = pct_data # what percent of training dataset is to be used
    model_name = 'M' + str(int(pct_data*100)) + 'L'

    models_path = './new_models/'
    model_save_path = models_path + model_name +'/'
    
    #### Do NOT touch these hyperparameters
    img_resize = 64
    depth = 3 # Depth of U-Net
    val_pct, test_pct = 0.2, 0.1 # Validation and test set %. 

    # Trainin params
    batch_size = 64
    epochs = 100
    lr = 1e-3 #Optimizer params

    # Use GPU if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f'Using device: {device}')
    
    
    # Initialize models, losses and optimizers. Make sure to re-run this cell again if you stop training and start again
    #Create 1 network
    modelS = model_UNet.UNet(in_channels=3, num_classes=2, depth=depth)
    modelS= modelS.to(device)

    #optimizer
    optimizer = Adam(modelS.parameters(), lr=lr)
    
    # Create dataloaders
    # Note: mixed_train_loader only contains labelled data due to is_mixed=False
    print('\nLOADING DATALOADER - can take a 1-5 mins to download and process data')
    mixed_train_loader, val_loader, test_loader = get_data(0.1,0.9,val_pct, test_pct, batch_size=batch_size, img_resize=img_resize, is_mixed_loader = False, pct_data = pct_data)
    print('DATALOADER CREATION COMPLETE')
    
    
    # ********************
    # START TRAINING
    # ********************
    print(f'*********************\nSTARTING TRAINING: Model {model_name}')
    
    # Create model save folder if not created
    if not(os.path.exists(models_path)):
        os.mkdir(models_path)
    if not(os.path.exists(model_save_path)):
        os.mkdir(model_save_path)
    
    eval_freq = 5
    losses, accsTr, IousTr, accsVal, IousVal = [], [], [], [], []

    for epoch in range(epochs):

            modelS.train()
            running_loss = 0

            for step, data in enumerate(mixed_train_loader):

                imgs, labs = data
                # Augment images
                imgS_aug = augmentation(imgs)

                imgS_aug = imgS_aug.to(device)
                labs = labs.squeeze().type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                # Forward pass for student and teacher
                z = modelS(imgS_aug) 
                loss = dice_loss(z, labs)
                
                loss.backward()
                
                optimizer.step()    
                running_loss += loss.item()


            print(f'Epoch: {epoch + 1:4d} - Loss: {running_loss:6.2f}')
            losses.append(running_loss)
        
            if (epoch % eval_freq == 0):

                # Get accuracy and IOU for train and validation dataset 
                accTr, IouTr = evaluate_model(modelS, mixed_train_loader, device)
                accVal, IouVal = evaluate_model(modelS, val_loader, device)
                
                accsTr.append(accTr)
                IousTr.append(IouTr)   
                accsVal.append(accVal)
                IousVal.append(IouVal)

                print(f'For training: accuracy-{accTr:2.0%}; IOU-{IouTr:2.0%}')
                print(f'For validation: accuracy-{accVal:2.0%}; IOU-{IouVal:2.0%}')

                np.savetxt(f"{model_save_path}running_loss", losses)
                np.savetxt(f"{model_save_path}train_accuracy", accsTr)
                np.savetxt(f"{model_save_path}train_IOU", IousTr)
                np.savetxt(f"{model_save_path}val_accuracy", accsVal)
                np.savetxt(f"{model_save_path}val_IOU", IousVal)

                # save model in cpu mode regardless of device
                modelS.to('cpu')
                torch.save(modelS.state_dict(), f"{model_save_path}model_epoch_{epoch+1}" + '.pt')
                modelS.to(device)
                
    print(f'TRAINING COMPLETE: Model {model_name}')
