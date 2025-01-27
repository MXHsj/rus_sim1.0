# =======================================================================
# file name:    vis.py
# description:  utility functions for visualization
# authors:      Xihan Ma, Mingjie Zeng
# date:         2023-02-25
# version:
# =======================================================================
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')   # turn off display to avoid multi-threading error

unload = transforms.ToPILImage()


def array2tensor(array: np.ndarray, device: torch.device = torch.device('cpu')) -> torch.tensor:
  ''' convert input image in numpy array to tensor
  :param array:   input image (W x H)
  :param device:  "cpu" / "cuda"
  :return:        image in tensor that can be directly fed into network (1 x 1 x W x H)
  '''
  array = np.expand_dims(array, axis=0)
  array = np.expand_dims(array, axis=0)
  tensor = torch.from_numpy(array).to(device).type(torch.float32)
  return tensor


def tensor2array(tensor: torch.Tensor, device: torch.device = torch.device('cpu')) -> np.array:
  ''' convert input image in tensor to numpy array
  :param tensor:  input image (1 x 1 x W x H)
  :param device:  "cpu" / "cuda"
  :return:        image in numpy array that can be directly displayed (W x H)
  '''
  if device.type == 'cpu':
    array = tensor.cpu().clone().detach().numpy()
  elif device.type == 'cuda':
    array = tensor.clone().detach().numpy()
  else:
    print(f'invalid device: {device}')
  array = np.squeeze(array)
  return array


def tensor2PIL(tensor: torch.Tensor, device: torch.device = torch.device('cuda')) -> Image:
  ''' convert input array in tensor to PIL Image
  :param:
  :param:
  :return: image in PIL Image that can be directly displayed (W x H)
  '''
  if device.type == 'cpu':
    image = tensor.cpu().clone()
  elif device.type == 'cuda':
    image = tensor.cuda().clone()
  else:
    print(f'invalid device: {device}')
  image = image.squeeze(0)
  image = unload(image)
  return image


def plot_segmentation(image, mask, pred_mask, acc=None):
  fig = plt.figure(figsize=(10, 4))
  rows = 1
  columns = 3

  fig.add_subplot(rows, columns, 1)
  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.title('input')

  fig.add_subplot(rows, columns, 2)
  plt.imshow(mask, cmap='gray')
  plt.axis('off')
  plt.title('ground truth')

  fig.add_subplot(rows, columns, 3)
  plt.imshow(pred_mask, cmap='gray')
  plt.axis('off')
  if acc is not None:
    plt.title(f'predicted, acc: {acc:.4f}')
  else:
    plt.title('predicted')

  # # find contours
  # mask_cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # pred_mask_cont, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # # draw contours
  # image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
  # image_mask_cont = image.copy()
  # image_pred_mask_cont = image.copy()
  # cv2.drawContours(image_mask_cont, mask_cont, -1, (0, 255, 0), 2)
  # cv2.drawContours(image_pred_mask_cont, pred_mask_cont, -1, (0, 255, 0), 2)

  # # draw to plt
  # fig.add_subplot(rows, columns, 4)
  # plt.imshow(image_mask_cont)
  # plt.axis('off')
  # plt.title('ground truth contour')

  # fig.add_subplot(rows, columns, 6)
  # plt.imshow(image_pred_mask_cont)
  # plt.axis('off')
  # plt.title('predicted contour')
  
  return fig
