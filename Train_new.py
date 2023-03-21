from preprocessing_1_dataloader import get_data
from unet import UNet
from utils import softmax_mse_loss, wt, update_ema_variables

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

if __name__ == '__main__':

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    epochs, batch_size = 10, 16
    ramp_up, consistency, alpha = 10, 56, 0.999

    # Import data loaders
    mixed_train_loader, val_loader, test_loader = get_data(0.2, 0.8, 0.1, 0.1, batch_size)

    # Initialize student and teacher networks 
    modelS = UNet(3,2).to(device) 
    modelT = UNet(3,2).to(device)

    sup_crit = nn.CrossEntropyLoss().to(device) # Supervised loss criterion
    optimizer = Adam(modelS.parameters()) # Student optimizer

    # Init global step
    global_step = 0

    print(f"\nInit all params, note we are running in {device}")

    # Start training loop
    for epoch in range(epochs):

        running_loss = 0

        for img, lab in mixed_train_loader:

            img, lab = img.to(device) , lab.to(device)

            optimizer.zero_grad()

            # Forward pass for student and teacher
            z, z_bar = modelS(img), modelT(img)

            # Find img with label
            sup_idx = torch.tensor([(elem != -1).item() for elem in lab[:, 0, 0, 0]]).to(device) #If batchsize is the first dim
            
            # Calculate losses
            Ls = sup_crit(torch.argmax(z[sup_idx], dim=1), lab[sup_idx].reshape(-1, 256, 256))
            Lu = softmax_mse_loss(z, z_bar).to(device)

            loss = Ls + wt(ramp_up, epoch, consistency) * Lu
            loss.backward()
            optimizer.step()    
            global_step += 1
            update_ema_variables(modelS, modelT, alpha, global_step)

            running_loss += loss.item()

            print('\nDone with one batch: ', loss.item())
        
        if epoch == 0:
                print('\n{:<10s}{:<10s}'.format('', 'Running loss')) 
        print('-'*25 +'\n' + '{:<10s}{:<10s}'.format(f"Epoch {epoch + 1}", f"{running_loss:.2f}%"))

        
    
    


