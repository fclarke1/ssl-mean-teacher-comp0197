import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets.utils as utils

#Weigth coef for the Unsupervised
def dice_loss(logits, targets, is_modelT_preds=True): 
  if is_modelT_preds:
    targets_animal = torch.unsqueeze(targets, dim=1)
  targets_animal = torch.squeeze(targets).to(torch.float32)
  preds_animal = F.softmax(logits, dim=1)
  preds_animal = preds_animal[:,1,:,:].to(torch.float32)
  eps = 1e-6
  intersection = (preds_animal * targets_animal).sum()
  dice_coef = (2. * intersection + eps) / ((preds_animal**2).sum() + (targets_animal**2).sum() + eps)
  dice_loss = 1 - dice_coef
  return dice_loss

@torch.no_grad()
def wt(rampup_length, current, alpha, wait_period = 5):

  if current < wait_period:
    return 0.0
    
  else:
    if rampup_length == 0:
                return 1.0
    else:
        current -= wait_period
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(alpha * np.exp(-5.0 * phase * phase))


#update the Teacher weight
@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step): 
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def unsup_loss(z, modelT, imgs, device): 

  #imgT_aug = augmentation(imgs).type(torch.float32)
  imgT_aug = imgs
  imgT_aug = imgT_aug.to(device)

  # z_bar will act as pseudo-labels
  with torch.no_grad():
    z_bar = modelT(imgT_aug)
    # z_bar = F.softmax(z_bar, dim = 1)
    z_bar_preds = torch.argmax(z_bar, dim=1)

  # Transform z_bar into predictions
  # Lu = unsup_crit(z, z_bar_preds).to(device)
  Lu = dice_loss(z, z_bar_preds, is_modelT_preds=True)

  return Lu

@torch.no_grad()
def evaluate_model(model, dataloader, device):
  
  model.eval()
  intersection_total, union_total = 0, 0
  pixel_correct, pixel_count = 0, 0
    
  for data in dataloader:
    imgs, labels = data
    imgs, labels = imgs.to(device), labels.to(device)
    logits = model(imgs)
    preds = torch.argmax(logits, dim=1)
    targets = torch.squeeze(labels)
            
    intersection_total += torch.logical_and(preds, targets).sum()
    union_total += torch.logical_or(preds, targets).sum()
            
    pixel_correct += (preds == targets).sum()
    pixel_count += targets.numel()

  iou = (intersection_total / union_total).item()
  accuracy = (pixel_correct / pixel_count).item()
  
  model.train()
  return accuracy, iou


