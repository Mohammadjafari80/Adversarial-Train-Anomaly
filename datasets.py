from torchvision.datasets import CIFAR10, MNIST, SVHN, CIFAR100, FashionMNIST
import os
import torch
import numpy as np

cifar_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
mnist_labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

def get_dataloader(transform, dataset='cifar', normal_class_indx = 0):
    pass


def get_CIFAR10(transform, normal_class_indx, batch_size):
    trainset = CIFAR10(root=os.path.join('~', 'cifar10'), train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = CIFAR10(root=os.path.join('~/', 'cifar10'), train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader



def get_MNIST(transform, normal_class_indx, batch_size):
    
    trainset = MNIST(root=os.path.join('~', 'mnist'), train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = MNIST(root=os.path.join('~', 'mnist'), train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_FASHION_MNIST(transform, normal_class_indx, batch_size):

    trainset = FashionMNIST(root=os.path.join('~', 'fashion-mnist'), train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]
    trainset.targets  = [0 for t in trainset.targets]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = FashionMNIST(root=os.path.join('~', 'fashion-mnist'), train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader





