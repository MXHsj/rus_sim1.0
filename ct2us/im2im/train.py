# =======================================================================
# file name:    train.py
# description:  train network
# authors:      Xihan Ma, Mingjie Zeng
# date:         2022-02-27
# version:
# =======================================================================
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from unet.model import UNet, AttentionUNet, UNet_dilation
from utils.data_loader import ConcatenateDataset, SingleSubjectDataset, RandomGenerator
from utils.predict import predict
from utils.evaluate import evaluate_mse, save_eval

from utils.vis import tensor2array

torch.manual_seed(42)

dir_checkpoint = Path('./checkpoints/')
dir_trainlog = Path('./training_log/')
dir_testlog = Path('./testing_log/')
dir_savedmodel = Path('./model/')
Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
Path(dir_trainlog).mkdir(parents=True, exist_ok=True)
Path(dir_testlog).mkdir(parents=True, exist_ok=True)
Path(dir_savedmodel).mkdir(parents=True, exist_ok=True)


def dataset(val_percent: float = 0.1,
            img_scale: float = None):
  ''' create training set and validation set
  @param val_percent:
  '''
  # random_generator = RandomGenerator()  # for data augmentation
  random_generator = None

  subjectList = [0]
  dataset = ConcatenateDataset(subjectID=subjectList, img_scale=img_scale, transform=random_generator)
  n_val = int(len(dataset) * val_percent)
  n_train = len(dataset) - n_val
  train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

  return train_set, val_set


def train_net(net, train_set: Dataset, val_set: Dataset, device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              save_checkpoint: bool = True,
              amp: bool = False):

  n_train = len(train_set)
  n_val = len(val_set)

  loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

  print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}''')

  # ========== Set up the optimizer, loss, learning rate scheduler, k-fold ==========
  # L2_reg = 1e-6  # 1e-8
  # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=L2_reg, momentum=0.9)
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4) ###
  # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(epochs/10), gamma=0.8)
  grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

  loss_mse = nn.MSELoss()
  loss_ce = nn.CrossEntropyLoss()
  loss_hu = nn.HuberLoss()
  loss_w = 1.0

  global_step = 0

  val_score_rec = []
  loss_dice_rec = []
  loss_ce_rec = []

  # ========== Begin training ==========
  for epoch in range(1, epochs+1):
    net.train()
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
      for batch in train_loader:
        images = batch['image']
        masks_true = batch['mask']

        assert images.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {images.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        images = images.to(device=device)
        masks_true = masks_true.to(device=device)

        # print(f'input shape: {images.shape}')
        # cv2.imwrite('input.png', 255*tensor2array(images))

        # print(f'output gt shape: {masks_true.shape}')
        # cv2.imwrite('gt.png', 255*tensor2array(masks_true))

        with torch.cuda.amp.autocast(enabled=amp):
          prob_pred = net(images)
            
            # print(f'prediction: {prob_pred.shape}, avg val: {prob_pred.mean()}, min val: {prob_pred.min()}')
            # cv2.imwrite('pred.png', 255*tensor2array(prob_pred))

          # ===== use ce to supervise =====
          mse = loss_mse(prob_pred, masks_true)
          ce = loss_ce(prob_pred, masks_true)
          # cossim = loss_cossim(prob_pred, masks_true)
          huber = loss_hu(prob_pred, masks_true)
          # loss = loss_w*mse + (1-loss_w)*ce
          loss = loss_w*huber
          # ==================================================

          # print(f'CE loss: {loss_CE}, dice loss: {loss_dice}, total loss: {loss}')

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        pbar.update(images.shape[0])
        global_step += 1
        pbar.set_postfix(**{'loss (batch)': loss.item()})

        # ===== Evaluation round =====
        division_step = (n_train // (10 * batch_size))
        if division_step > 0:
          if global_step % division_step == 0:
            tag = 'epoch_' + str(epoch) + '_step_' + str(global_step)

            save_eval(images[0], masks_true[0], prob_pred, dir=dir_trainlog, tag=tag)
              
            val_score, val_score_std = evaluate_mse(net, val_loader, device)

            # scheduler.step(val_score)
            # scheduler.step()

            print(f'''Validation info:
                  Learning rate: {optimizer.param_groups[0]['lr']}
                  Validation Score: {val_score:.4f} ± {val_score_std:.4f}
                  Step: {global_step}
                  Epoch: {epoch}''')
            loss_ce_rec.append(loss.item())
            val_score_rec.append(val_score.item())

    if save_checkpoint:
      torch.save(net.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
      print(f"Checkpoint {epoch} saved!")

  fig = plt.figure(figsize=(10, 7))
  plt.plot(np.arange(len(val_score_rec)), val_score_rec, label='validation score')
  plt.plot(np.arange(len(loss_dice_rec)), loss_dice_rec, label="loss")
  plt.legend()
  plt.savefig(str(dir_trainlog / 'loss_curve.png'))
  plt.close(fig)


def test(net, test_set, device, saveDir=dir_testlog):
  ''' test model accuracy on unseen dataset
  '''
  loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
  test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
  test_score, test_score_std = evaluate_mse(net, test_loader, device, saveDir=dir_testlog)
  print(f"testing score: {test_score:.4f} ± {test_score_std:.4f}")


def get_args():
  parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
  parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
  parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
  parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=5e-4, help='Learning rate', dest='lr')
  parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
  parser.add_argument('--scale', '-s', type=float, default=-1, help='Downscaling factor of the images')
  parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0, help='Percentage for validation')
  parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
  parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
  parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
  return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device {device}")

  # net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
  net = AttentionUNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
  # net = UNet_dilation(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
  
  print(f'''Network:\n
        \t{net.n_channels} input channels\n
        \t{net.n_classes} output channels (classes)\n
        \t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling''')

  if args.load:
    net.load_state_dict(torch.load(args.load, map_location=device))
    print(f"Model loaded from {args.load}")


  net.to(device=device)
  train_set, val_set = dataset(val_percent=args.val/100, img_scale=args.scale)

  try:
    train_net(net=net,
              train_set=train_set,
              val_set=val_set,
              epochs=args.epochs,
              batch_size=args.batch_size,
              learning_rate=args.lr,
              device=device,
              amp=args.amp)

    test(net=net, test_set=val_set, device=device)

  except KeyboardInterrupt:
    torch.save(net.state_dict(), str(dir_checkpoint/'INTERRUPTED.pth'))
    print(f"Saved interrupt")
    raise
