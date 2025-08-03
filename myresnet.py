import torch, torchvision, torchvision.transforms as transforms
import torch.nn as nn
import logging, time
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.4465],
                         std = [0.202, 0.1994, 0.2010])
])

trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download = True)
testset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform, download = True)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size = 128, shuffle = True, num_workers = 2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size = 128, shuffle = True, num_workers = 2
)

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
    residual = x;
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      residual = self.downsample(x)
    out = self.relu(out + residual)

    return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes = 10):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(16)
    )
    self.layer_group1 = self.make_layer_group(block, 16, layers[0], stride = 1)
    self.layer_group2 = self.make_layer_group(block, 16, layers[1], stride = 2)
    self.layer_group3 = self.make_layer_group(block, 32, layers[2], stride = 2)
    self.fc = nn.Linear(64, num_classes)

  @staticmethod
  def global_avg_pool(x):
    pool = nn.AdaptiveAvgPool2d((1,1))
    pooled = pool(x)
    pooled = pooled.view(pooled.shape[0], -1)

    return pooled

  def make_layer_group(self, block, in_channels, layer_group_size, stride):
    downsample = None
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
    out = self.conv1(x)
    out = self.layer_group1(out)
    out = self.layer_group2(out)
    out = self.layer_group3(out)
    out = self.global_avg_pool(out)
    out = self.fc(out)

    return out