from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim

class FeatureExtractor(nn.Module):
  def __init__(self, model=18, pretrained=True):
    super(FeatureExtractor, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = None

    if model == 18:
        self.model = models.resnet18(pretrained=pretrained)
    else:
        self.model = models.resnet152(pretrained=pretrained)

    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Flatten()
    self.head = nn.Linear(num_ftrs, 2)

  def forward(self, x):
    x = self.model(x)
    x = self.head(x)
    return x

  def get_feature_vector(self, x):
    x = self.model(x)
    return x