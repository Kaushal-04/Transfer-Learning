# 1st
# torchvision package provides access to popular datasets
# pip install torchvision
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_data(batch_size, data_dir="data"):
    '''Load the Fashion-MNIST dataset'''

    # define transfer to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    # Take the range of pixel value and shift them : Normalize

    # Download and load the training data
    trainset = datasets.FashionMNIST(data_dir, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    testset = datasets.FashionMNIST(data_dir, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader

trainloader, testloader = load_data(64)
# Batch Size = 64