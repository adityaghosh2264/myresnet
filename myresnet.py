import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels)
    )
    self.downsample = downsample
    self.out_channels = out_channels
    self.relu = nn.ReLU()

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      residual = self.downsample(x)
    out = self.relu(out + residual)

    return out

class ResNet(nn.Module):
  def __init__(self, block, n, num_classes = 10):
    super().__init__()
    layers = [2*n]*3
    start = 16
    conv1 = nn.Sequential(
        nn.Conv2d(3, start, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(start)
    )
    num_layers = len(layers)
    self.layer_flow = [conv1]
    self.layer_flow.append(self.make_layer_group(block, start, layers[0], stride = 1))
    for i in range(1, num_layers):
        self.layer_flow.append(self.make_layer_group(block, start, layers[i], stride = 2))
        start *= 2
    self.fc = nn.Linear(start, num_classes)

    self.layer_flow = nn.Sequential(*self.layer_flow)

  @staticmethod
  def global_avg_pool(x):
    pool = nn.AdaptiveAvgPool2d((1,1))
    pooled = pool(x)
    pooled = pooled.view(pooled.shape[0], -1)

    return pooled
  @staticmethod
  def make_layer_group(block, in_channels, layer_group_size, stride):
    layers = []
    if stride != 1:
      downsample = nn.Sequential(
          nn.Conv2d(in_channels, 2*in_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
          nn.BatchNorm2d(2*in_channels)
      )
      layers.append(block(in_channels, 2 * in_channels, stride, downsample))
      for i in range(1, layer_group_size):
        layers.append(block(2*in_channels, 2*in_channels, stride = 1))
    else:
      for i in range(layer_group_size):
        layers.append(block(in_channels, in_channels))

    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.layer_flow(x)
    out = self.global_avg_pool(out)
    out = self.fc(out)

    return out
