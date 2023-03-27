import torch
import numpy as np
@torch.no_grad()
def wt(rampup_length, current, alpha, wait_period = 5):

  if current < wait_period:
    return 0.0
    
  else:
    if rampup_length == 0:
                return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(alpha * np.exp(-5.0 * phase * phase))
#update the Teacher weigth
@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step): 
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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