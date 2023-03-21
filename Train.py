import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from unet import UNet
from preprocessing_1_dataloader import get_data
from data_augmentation import augmentation, colorjiter, invert
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable GPU support
####Initialisation####
#create 2 network
Student = UNet(3,2).to(device)
Teacher = UNet(3,2).to(device)
#creat the losses
sup_crit = nn.CrossEntropyLoss().to(device)
unsup_crit = nn.CrossEntropyLoss().to(device)
#optimizer
optimizer = Adam(Student.parameters())
#other HP
batch_size = 64
epochs = 16
ramp_up = 10
consistency = 56
alpha = 0.999
gs = 0

#Weigth coef for the Unsupervised
def wt(rampup_length, current, alpha):
    if rampup_length == 0:
                return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(alpha * np.exp(-5.0 * phase * phase))
#update the Teacher weigth
def update_ema_variables(model, ema_model, alpha, global_step): 
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


##data loader
mixed_train_loader, val_loader, test_loader = get_data(0.2,0.8,0.2,0.1)
#Train
for epoch in range(epochs):
    cum_loss = 0
    for idx, (X,y) in enumerate(mixed_train_loader):
        optimizer.zero_grad()
        X_student_augmentation = augmentation(X)
        X_teacher_augmentation = augmentation(X)
        pred_stud = Student(X_student_augmentation)
        pred_teach = Teacher(X_teacher_augmentation)

        # Find img with label
        idx = [elem != -1 for elem in y[:, 0, 0, 0]] #If batchsize is the first dim

        # Calculate supervised and unsupervised losses
        Ls = sup_crit(pred_stud[idx], y[idx])
        Lu = unsup_crit(pred_stud, pred_teach)
        loss = Ls + Lu * wt(ramp_up, epoch, consistency)
        loss.backward()
        optimizer.step()
        cum_loss += loss
        gs += 1
        update_ema_variables(Student, Teacher, alpha, gs)
    print(f'Epoch {epoch} loss {cum_loss.item() / len(mixed_train_loader)}')
        #get the images with mask
        #for i in range(batch_size): #loop through the label in the batch
         #   if y[0][0][0][i] != None: # check if the mask is a real one or a fake one
                

