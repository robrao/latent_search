import torch

from torchvision import datasets, transforms

if __name__ == '__main__':
    mnist = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())