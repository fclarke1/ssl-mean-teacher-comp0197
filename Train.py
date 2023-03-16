import torch
import torch.nn as nn
from torch.optim import Adam
from unet import unet_model as Unet

####Initialisation####
#create 2 network
Student = Unet()
Teacher = Unet()
#creat the losses
sup_crit = nn.CrossEntropyLoss()
unsup_crit = nn.CrossEntropyLoss()
#optimizer
optimizer = Adam(Student.parameters())
#other HP
batch_size = 64
epochs = 16