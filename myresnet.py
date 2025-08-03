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

