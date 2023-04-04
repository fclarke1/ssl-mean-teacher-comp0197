import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import model_UNet
from data_augmentation import augmentation
from data_into_loaders import get_data
from utils import dice_loss, wt, update_ema_variables, unsup_loss, evaluate_model



def mean_teacher_train(supervised_pct):
    """
    Given the percentage of training dataset being labelled (the remaining train dataset will be unlabelled)
    this function trains a UNet model using SSL mean teacher algorithm
    The model trained is saved in the folder "/models/model_name"
    
    supervised_pct (float): percentage of the training dataset being labelled in (0,1]
    """

    # ONLY PARAMETER THAT SHOULD BE CHANGED. 
    supervised_pct = supervised_pct # what percent of training is to be labelled
    model_name = 'M' + str(int(supervised_pct*100))

    models_path = './new_models/'
    model_save_path = models_path + model_name +'/'

    # Do NOT change these hyperparams 
    img_resize = 64   
    val_pct, test_pct = 0.2, 0.1 # Validation and test set %. 

    depth = 3  # U-Net params

    # Training params
    epochs, lr = 100, 1e-3
    batch_size = 32

    ramp_up, consistency, wait_period = 25, 1.5, int(1000/161 * 1.1) # Teacher contribution params
    alpha = 0.999 # Teacher update params
    global_step = 0

    # Use GPU if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')


    #### Initialisation ####
    #create 2 network
    modelS = model_UNet.UNet(in_channels=3, num_classes=2, depth=depth)
    modelT = model_UNet.UNet(in_channels=3, num_classes=2, depth=depth)
    modelS,  modelT= modelS.to(device), modelT.to(device)

    #optimizer
    optimizer = Adam(modelS.parameters(), lr=lr)

    # Create the dataloaders
    # mixed_train_loader contains labelled and unlabelled data
    print('\nLOADING DATALOADER - can take a 1-5 mins to download and process data')
    mixed_train_loader, val_loader, test_loader = get_data(supervised_pct,1 - supervised_pct, val_pct, test_pct, batch_size=batch_size, img_resize=img_resize)
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
    losses, sup_losses, unsup_losses, accsTr, IousTr, accsVal, IousVal, wts = [], [], [], [], [], [], [] , []

    modelS.train() # Put model in train mode

    for epoch in range(epochs):

            running_loss, running_loss_sup, running_loss_unsup = 0, 0, 0 # Init running losses
            w_t = wt(rampup_length=ramp_up, current=epoch, alpha=consistency, wait_period=wait_period) # Get unsupervised weight for the epoch

            for step, data in enumerate(mixed_train_loader):

                imgs, labs = data
                imgS_aug = augmentation(imgs) # Augment images

                imgS_aug = imgS_aug.to(device)
                labs = labs.squeeze().type(torch.LongTensor).to(device)

                optimizer.zero_grad()

                # Forward pass for student and teacher
                z = modelS(imgS_aug) 

                sup_idx = torch.tensor([(elem != -1).item() for elem in labs[:, 0, 0]]).to(device) #If batchsize is the first dim

                Ls = dice_loss(z[sup_idx], labs[sup_idx])   # supervised loss
                Lu = unsup_loss(z, modelT, imgs, device)    # unsupervised loss
                loss = Ls + w_t * Lu                        # total loss
                
                loss.backward()
                
                optimizer.step()    
                global_step += 1
                update_ema_variables(modelS, modelT, alpha, global_step)    # update teacher model
                running_loss += loss.item()
                running_loss_sup += Ls.item()
                running_loss_unsup += Lu.item()
        
            # Print update every epoch
            print(f'Epoch: {epoch + 1:4d} - Loss: {running_loss:6.2f}, loss_sup: {running_loss_sup:6.1f}, loss_unsup: {running_loss_unsup:6.1f}, w_t: {w_t: 3.2f}')
            losses.append(running_loss)
            sup_losses.append(running_loss_sup)
            unsup_losses.append(running_loss_unsup)
            wts.append(w_t)

            # Log evaluation metrics every eval_freq epochs
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
                np.savetxt(f"{model_save_path}sup_loss", sup_losses)
                np.savetxt(f"{model_save_path}unsup_loss", unsup_losses)

                np.savetxt(f"{model_save_path}train_acc", accsTr)
                np.savetxt(f"{model_save_path}train_IOU", IousTr)
                np.savetxt(f"{model_save_path}val_acc", accsVal)
                np.savetxt(f"{model_save_path}val_IOU", IousVal)

                np.savetxt(f"{model_save_path}weights", wts)

                # save model in cpu mode regardless of device
                modelS.to('cpu')
                torch.save(modelS.state_dict(), f"{model_save_path}student_epoch_{epoch+1}" + '.pt')
                modelS.to(device)


    print(f'TRAINING COMPLETE: Model {model_name}')