import torch
import torch.nn.functional as F
def dice_loss(logits, targets): 
  preds_animal = F.softmax(logits, dim=1)
  targets_animal = torch.squeeze(targets)
  preds_animal = preds_animal[:,1,:,:]
  eps = 1e-6
  intersection = (preds_animal * targets_animal).sum()
  dice_coef = (2. * intersection + eps) / ((preds_animal**2).sum() + eps)
  dice_loss = 1 - dice_coef
  return dice_loss