from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
mnist_labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

#######################
#  Define Transform   #
#######################

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

transform_1_channel = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)



def get_dataloader(dataset='cifar', normal_class_indx = 0, batch_size=8):
    if dataset == 'cifar':
        return get_CIFAR10(normal_class_indx, batch_size)
    elif dataset == 'mnist':
        return get_MNIST(normal_class_indx, batch_size)
    elif dataset == 'fashion-mnist':
        return get_FASHION_MNIST(normal_class_indx, batch_size)
    else:
        raise Exception("Dataset is not supported yet. ")


def get_CIFAR10(normal_class_indx, batch_size):
    trainset = CIFAR10(root=os.path.join('~', 'cifar10'), train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root=os.path.join('~/', 'cifar10'), train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader



def get_MNIST(normal_class_indx, batch_size):
    
    trainset = MNIST(root=os.path.join('~', 'mnist'), train=True, download=True, transform=transform_1_channel)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = MNIST(root=os.path.join('~', 'mnist'), train=False, download=True, transform=transform_1_channel)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_FASHION_MNIST(normal_class_indx, batch_size):

    trainset = FashionMNIST(root=os.path.join('~', 'fashion-mnist'), train=True, download=True, transform=transform_1_channel)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = FashionMNIST(root=os.path.join('~', 'fashion-mnist'), train=False, download=True, transform=transform_1_channel)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


class Exposure(Dataset):
    def __init__(self, root):
        self.image_files = os.path.join(root, "*.png")

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_files)

