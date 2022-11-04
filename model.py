from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import preactresnet

from wideresnet import WideResNet

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()


class FeatureExtractor(nn.Module):
  def __init__(self, model=18, pretrained=True):
    super(FeatureExtractor, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = None
    self.norm = lambda x: ( x - mu ) / std
    
    if model == 18:
        self.model = preactresnet.PreActResNet18()
    elif model == 34:
        self.model = preactresnet.PreActResNet34()
    elif model == 50:
        self.model = preactresnet.PreActResNet50()
    elif model == 101:
        self.model = preactresnet.PreActResNet101()
    elif model == 152:
        self.model = preactresnet.PreActResNet152()
    else:
        model = WideResNet(34, 10, widen_factor=10, dropRate=0.0)

    num_ftrs = self.model.linear.in_features
    self.model.linear = nn.Flatten()
    self.head = nn.Linear(num_ftrs, 2)

  def forward(self, x):
    x = self.norm(x)
    x = self.model(x)
    x = self.head(x)
    return x

  def get_feature_vector(self, x):
    x = self.norm(x)
    x = self.model(x)
    return x



class FeatureExtractor(nn.Module):
  def __init__(self, model=18, pretrained=True):
    super(FeatureExtractor, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = None
    self.norm = lambda x: ( x - mu ) / std
    
    if model == 18:
        self.model = preactresnet.PreActResNet18()
    elif model == 34:
        self.model = preactresnet.PreActResNet34()
    elif model == 50:
        self.model = preactresnet.PreActResNet50()
    elif model == 101:
        self.model = preactresnet.PreActResNet101()
    else:
        self.model = preactresnet.PreActResNet152()

    num_ftrs = self.model.linear.in_features
    self.model.linear = nn.Flatten()
    self.head = nn.Linear(num_ftrs, 2)

  def forward(self, x):
    x = torch.clamp(x + torch.randn(x.size()).cuda() * 0.2, min=0, max=1)
    x = self.norm(x)
    x = self.model(x)
    x = self.head(x)
    return x

  def get_feature_vector(self, x):
    x = torch.clamp(x + torch.randn(x.size()).cuda() * 0.2, min=0, max=1)
    x = self.norm(x)
    x = self.model(x)
    return x


net = FeatureExtractor(model='wide')
print(net)