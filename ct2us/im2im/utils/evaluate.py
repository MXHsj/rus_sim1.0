# =======================================================================
# file name:    evaluate.py
# description:  evaluate dice score
# authors:      Xihan Ma, Mingjie Zeng
# date:         2023-02-25
# version:
# =======================================================================
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.vis import tensor2array, plot_segmentation
from utils.predict import predict

def evaluate_mse(net, dataloader, device, saveDir: Path = None):
  ''' evaluate binary cross entropy score
  @param net:
  @param dataloader:
  @param device:
  @param saveDir: path to save segmentation results
  '''
  net.eval()
  num_val_batches = len(dataloader)
  bce_score_rec = []
  criterion = nn.MSELoss()
  for itr, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
    image, mask_true = batch['image'], batch['mask']
    image = image.to(device=device, dtype=torch.float32)
    mask_true = mask_true.to(device=device, dtype=torch.float32)

    with torch.no_grad():
      prob_pred = predict(image, net, device=device)

      # compute MSE
      bce_score = criterion(prob_pred, mask_true).item()
      bce_score_rec.append(bce_score)

      # if saveDir is not None:
        # save_eval(image[0], mask_true[0], mask_pred_raw, dir=saveDir, tag='eval_'+str(itr), score=dice_score)

  net.train()
  bce_score_rec = torch.tensor(bce_score_rec, device=device, dtype=torch.float32)
  if num_val_batches == 0:
    return bce_score
  return torch.mean(bce_score_rec), torch.std(bce_score_rec)

def evaluate_dice(net, dataloader, device, saveDir: Path = None):
  ''' evaluate dice score
  @param net:
  @param dataloader:
  @param device:
  @param saveDir: path to save segmentation results
  '''
  net.eval()
  num_val_batches = len(dataloader)
  dice_score_rec = []
  for itr, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
    image, mask_true_raw = batch['image'], batch['mask']
    image = image.to(device=device, dtype=torch.float32)
    mask_true_raw = mask_true_raw.to(device=device, dtype=torch.long)
    mask_true = F.one_hot(mask_true_raw, net.n_classes).permute(0, 3, 1, 2).float()

    with torch.no_grad():
      # mask_pred_raw = net(image)
      # if net.n_classes == 1:
      #   mask_pred = (F.sigmoid(mask_pred_raw) > 0.5).float()
      #   # compute the Dice score, epsilon: smooth value
      #   dice_score = dice_coeff(mask_pred, mask_true, reduce_batch_first=False, epsilon=1)
      # else:
      #   mask_pred = F.one_hot(mask_pred_raw.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
      #   # compute the Dice score, ignoring background
      #   dice_score = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False, epsilon=1)
      mask_pred_raw, prob_pred = predict(image, net, device=device)
      mask_pred = mask_pred_raw.float()
      # compute the Dice score, ignoring background
      dice_score = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False, epsilon=1)

      dice_score_rec.append(dice_score)
      if saveDir is not None:
        save_eval(image[0], mask_true_raw[0], mask_pred_raw, dir=saveDir, tag='eval_'+str(itr), score=dice_score)

  net.train()
  dice_score_rec = torch.tensor(dice_score_rec, device=device, dtype=torch.float32)
  if num_val_batches == 0:
    return dice_score
  return torch.mean(dice_score_rec), torch.std(dice_score_rec)


def save_eval(image, mask_true, mask_pred, dir: Path = None, tag: str = None, score=None):
  ''' save segmentation results
  @param image: B-mode US
  @param mask_true: ground truth
  @param mask_pred: prediction
  @param dir: path to save segmentation results
  @param tag:
  @param score: dice score
  '''
  image = tensor2array(image.float())
  mask_true = tensor2array(mask_true.float())

  # TODO: plot multi-class segmentation
  # mask_pred = tensor2array(mask_pred.argmax(dim=1).float())
  mask_pred = tensor2array(mask_pred[0, 0, :, :].float())

  fig2save = plot_segmentation(image, mask_true, mask_pred, acc=score)
  if dir is not None and tag is not None:
    fig2save.savefig(str(dir / f'{tag}.png'))
    plt.close(fig2save)
