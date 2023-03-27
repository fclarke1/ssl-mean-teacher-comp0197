import torch
import torch.nn.functional as F
from data_augmentation import augmentation
def unsup_loss(z, modelT, imgs, unsup_crit, device): 

  imgT_aug = augmentation(imgs).type(torch.float32)
  imgT_aug = imgT_aug.to(device)

  # z_bar will act as pseudo-labels
  z_bar = modelT(imgT_aug)
  z_bar = F.softmax(z_bar, dim = 1)
  z_bar_preds = torch.argmax(z_bar, dim=1)

  # Transform z_bar into predictions
  Lu = unsup_crit(z, z_bar_preds).to(device)

  return Lu