# ===============================================================
# file name:    model.py
# description:  naive U-Net implementation
# author:       Xihan Ma, Mingjie Zeng
# date:         2022-11-12
# ===============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv_dilation(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, dilation=4, padding=4, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)
  
class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

class Down_dilation(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv_dilation(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)
  
class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)

class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    # if bilinear, use the normal convolutions to reduce the number of channels
    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    else:
      self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.out_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                  nn.Tanh())

  def forward(self, x):
    # return F.softmax(self.conv(x), dim=1)  # normalize class probabilities
    return self.out_conv(x)
    
class ChannelAttention(nn.Module):
  def __init__(self, in_planes, reduction_ratio=16):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    
    self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
    max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
    out = avg_out + max_out
    return self.sigmoid(out) * x

class UNet(nn.Module):
  '''
  U-Net
  '''
  def __init__(self, n_channels, n_classes, bilinear=False):
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.inc = DoubleConv(n_channels, 64)
    self.down1 = Down(64, 128)
    self.down2 = Down(128, 256)
    self.down3 = Down(256, 512)
    factor = 2 if bilinear else 1
    self.down4 = Down(512, 1024 // factor)
    self.up1 = Up(1024, 512 // factor, bilinear)
    self.up2 = Up(512, 256 // factor, bilinear)
    self.up3 = Up(256, 128 // factor, bilinear)
    self.up4 = Up(128, 64, bilinear)
    self.outc = OutConv(64, n_classes)

  def forward(self, x):
    # print(f"input type: {type(x)}")
    x1 = self.inc(x)
    # print(f"x1 type: {type(x1)}")
    x2 = self.down1(x1)
    # print(f"x2 type: {type(x2)}")
    x3 = self.down2(x2)
    # print(f"x3 type: {type(x3)}")
    x4 = self.down3(x3)
    # print(f"x4 type: {type(x4)}")
    x5 = self.down4(x4)
    # print(f"x5 type: {type(x5)}")
    x = self.up1(x5, x4)
    # print(f"x up1 type: {type(x)}")
    x = self.up2(x, x3)
    # print(f"x up2 type: {type(x)}")
    x = self.up3(x, x2)
    # print(f"x up3 type: {type(x)}")
    x = self.up4(x, x1)
    # print(f"x up4 type: {type(x)}")
    logits = self.outc(x)
    # print(f"out type: {type(x)}")
    # print(f"out: {logits}")
    return logits

class AttentionUNet(nn.Module):
  '''
  U-Net with channel attention
  '''
  def __init__(self, n_channels, n_classes, bilinear=False):
    super(AttentionUNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.inc = DoubleConv(n_channels, 64)
    self.cam1 = ChannelAttention(64)
    self.down1 = Down(64, 128)
    self.cam2 = ChannelAttention(128)
    self.down2 = Down(128, 256)
    self.cam3 = ChannelAttention(256)
    self.down3 = Down(256, 512)
    self.cam4 = ChannelAttention(512)
    factor = 2 if bilinear else 1
    self.down4 = Down(512, 1024 // factor)
    self.cam5 = ChannelAttention(1024 // factor)
    self.up1 = Up(1024, 512 // factor, bilinear)
    self.up2 = Up(512, 256 // factor, bilinear)
    self.up3 = Up(256, 128 // factor, bilinear)
    self.up4 = Up(128, 64, bilinear)
    self.outc = OutConv(64, n_classes)

  def forward(self, x):
    # print(f"input type: {type(x)}")
    x1 = self.inc(x)
    x1_cam = self.cam1(x1)
    # print(f"x1 type: {type(x1)}")
    x2 = self.down1(x1_cam)
    x2_cam = self.cam2(x2)
    # print(f"x2 type: {type(x2)}")
    x3 = self.down2(x2_cam)
    x3_cam = self.cam3(x3)
    # print(f"x3 type: {type(x3)}")
    x4 = self.down3(x3_cam)
    x4_cam = self.cam4(x4)
    # print(f"x4 type: {type(x4)}")
    x5 = self.down4(x4_cam)
    x5_cam = self.cam5(x5)
    # print(f"x5 type: {type(x5)}")
    x = self.up1(x5_cam, x4_cam)
    # print(f"x up1 type: {type(x)}")
    x = self.up2(x, x3_cam)
    # print(f"x up2 type: {type(x)}")
    x = self.up3(x, x2_cam)
    # print(f"x up3 type: {type(x)}")
    x = self.up4(x, x1_cam)
    # print(f"x up4 type: {type(x)}")
    logits = self.outc(x)
    # print(f"out type: {type(x)}")
    # print(f"out: {logits}")
    return logits

class UNet_dilation(nn.Module):
  ''' 
  U-Net with dialation convolution layer for downsampling
  '''
  def __init__(self, n_channels, n_classes, bilinear=False):
    super(UNet_dilation, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.inc = DoubleConv(n_channels, 64)
    self.down1 = Down_dilation(64, 128)
    self.down2 = Down_dilation(128, 256)
    self.down3 = Down_dilation(256, 512)
    factor = 2 if bilinear else 1
    self.down4 = Down_dilation(512, 1024 // factor)
    self.up1 = Up(1024, 512 // factor, bilinear)
    self.up2 = Up(512, 256 // factor, bilinear)
    self.up3 = Up(256, 128 // factor, bilinear)
    self.up4 = Up(128, 64, bilinear)
    self.outc = OutConv(64, n_classes)

  def forward(self, x):
    # print(f"input type: {type(x)}")
    x1 = self.inc(x)
    # print(f"x1 type: {type(x1)}")
    x2 = self.down1(x1)
    # print(f"x2 type: {type(x2)}")
    x3 = self.down2(x2)
    # print(f"x3 type: {type(x3)}")
    x4 = self.down3(x3)
    # print(f"x4 type: {type(x4)}")
    x5 = self.down4(x4)
    # print(f"x5 type: {type(x5)}")
    x = self.up1(x5, x4)
    # print(f"x up1 type: {type(x)}")
    x = self.up2(x, x3)
    # print(f"x up2 type: {type(x)}")
    x = self.up3(x, x2)
    # print(f"x up3 type: {type(x)}")
    x = self.up4(x, x1)
    # print(f"x up4 type: {type(x)}")
    logits = self.outc(x)
    # print(f"out type: {type(x)}")
    # print(f"out: {logits}")
    return logits

if __name__ == '__main__':
  # test case
  net = UNet(n_channels=1, n_classes=1, bilinear=False)
  test_img = torch.rand((1, 1, 100, 100))
  test_out = net(test_img)
  print(f'test in: {test_img}\ntest out: {test_out}')