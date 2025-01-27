# =======================================================================
# file name:    data_loader.py
# description:  implement dataloader for pytorch
# authors:      Xihan Ma, Mingjie Zeng
# date:         2023-02-25
# version:
# =======================================================================
import os
import cv2
import random
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
from skimage.restoration import denoise_nl_means

INPUT_HEIGHT = 240    # 240
INPUT_WIDTH = 240     # 240


def random_flip(image: np.ndarray, label: np.ndarray):
  image = np.flip(image, axis=1).copy()  # horizontal flip
  label = np.flip(label, axis=1).copy()
  return image, label

def random_flip_vertical(image: np.ndarray, label: np.ndarray):
  image = np.flip(image, axis=0).copy()  # vertical flip
  label = np.flip(label, axis=0).copy()
  return image, label


def random_rotate(image: np.ndarray, label: np.ndarray):
  angle = np.random.randint(-10, 10)
  image = ndimage.rotate(image, angle, order=0, reshape=False)
  label = ndimage.rotate(label, angle, order=0, reshape=False)
  return image, label


def random_brightness(image: np.ndarray, label: np.ndarray):
  alpha = 2.*np.random.rand()  # Contrast control
  beta = np.random.randint(1, 5)  # Brightness control
  image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
  return image, label


class RandomGenerator():
  def __init__(self):
    pass

  def __call__(self, image, label):
    output_size = image.shape
    if random.random() > 0.5:
      image, label = random_flip(image, label)
    if random.random() > 0.5:
      image, label = random_rotate(image, label)
    # if random.random() > 0.6:
    #   image, label = random_flip_vertical(image, label)
    # if random.random() > 0.3:
    #   image, label = random_brightness(image, label)
    x, y = image.shape
    if x != output_size[0] or y != output_size[1]:
      image = cv2.resize(image, (output_size[0], output_size[1]))
      label = cv2.resize(label, (output_size[0], output_size[1]))
    return image, label


class SingleSubjectDataset(Dataset):
  ''' single subject dataset
  '''
  subjectList = {
      0: 'ct_slices_trans_06-Mar-2024'
  }

  def __init__(self, subjectID: int = 0,
               img_scale: float = -1,
               transform: RandomGenerator = None):
    '''
    @param subjectID: select ONE subject from subjectList
    @param transform: image transformation method for data augmentation
    '''
    self.img_scale = img_scale
    self.transform = transform
    self.subject = self.subjectList[subjectID]
    self.msk_dir = os.path.join(os.path.dirname(__file__), '../../dataset/'+self.subject+'/us/')
    self.img_dir = os.path.join(os.path.dirname(__file__), '../../dataset/'+self.subject+'/ct_msk/')

    # ========== sanity check: each feature should have same number of masks ==========

    # ========== get mask indices ==========
    self.msk_indices = []  # indices in the dataset with annotation
    for name in os.listdir(self.msk_dir):
      name_trimmed = name[0:-1-3]  # exclude file extention (3 characters)
      self.msk_indices.append(name_trimmed.split('_')[-1])
    self.msk_indices = sorted(self.msk_indices)

  def __len__(self):
    return len(self.msk_indices)

  def __getitem__(self, idx):
    msk_idx = self.msk_indices[idx]
    img_name = self.img_dir+'ctmsk_'+msk_idx+'.png'
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    msk_name = self.msk_dir+'bmode_'+msk_idx+'.png'
    msk = cv2.imread(msk_name, cv2.IMREAD_GRAYSCALE)

    if self.transform is not None:
      img, msk = self.transform(img, msk)
    img = self.preprocess(img)
    msk = self.preprocess(msk)
    return {'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(msk.copy()).float().contiguous()}

  def preprocess(self, input):
    ''' preprocess input image and mask
    '''
    # input_ = input[100:200, 150:250]  # debug
    input_ = input.copy()  # debug
    if self.img_scale == -1:
      input_resize = input_.copy()
    else:
      input_resize = cv2.resize(input_, (int(self.img_scale*INPUT_WIDTH), int(self.img_scale*INPUT_HEIGHT)))

    processed = input_resize.copy()

    # processed = denoise_nl_means(processed, patch_size=3, patch_distance=5, h=0.02)

    if processed.ndim == 2:
      processed = processed[np.newaxis, ...]
    else:
      processed = processed.transpose((2, 0, 1))
    processed = processed / 255

    return processed

  def view_item(self, idx):
    ''' create mask overlaid image
    '''
    ...


class ConcatenateDataset(Dataset):
  ''' training set concatenating multiple subjects' datasets
  '''

  def __init__(self, subjectID: list = [0],
               img_scale: float = -1,
               transform: RandomGenerator = None):
    # ===== instantiate individual subject datasets =====
    self.subdatasets = []
    for ID in subjectID:
      self.subdatasets.append(SingleSubjectDataset(subjectID=ID,
                                                   img_scale=img_scale,
                                                   transform=transform))
    # ===== count total number of images =====
    self.total_num = 0
    for dataset in self.subdatasets:
      self.total_num += len(dataset)

  def __len__(self):
    return self.total_num

  def __getitem__(self, idx):
    acc = 0
    for i, dataset in enumerate(self.subdatasets):
      if idx < len(dataset) + acc:
        return dataset[idx - acc]
      else:
        acc += len(dataset)


# ========== unit test ==========
if __name__ == '__main__':
  # test_data = SingleSubjectDataset(subjectID=0, featureID=[0])
  # print(len(test_data))
  # test_data[0]

  train_data = ConcatenateDataset(subjectID=[0], img_scale=-1)
  print(f'dataset size: {len(train_data)}')
  print(f"input ct size: {train_data[10]['image'].shape}, range: {train_data[10]['image'].min()}-{train_data[10]['image'].max()}")
  print(f"input us size {train_data[10]['mask'].shape}, range: {train_data[10]['mask'].min()}-{train_data[10]['mask'].max()}")
