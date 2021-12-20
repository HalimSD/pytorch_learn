import torch
import torchvision
import torchvision.transforms as transforms


# for normalization 
            #MEANS          #DEVIATIONS 
statistics = ((0.5,0.5,0.5),(0.5,0.5,0.5))
dataset_transformed = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #transform images to tensors with CHW
    transforms.Normalize(*statistics, inplace=True) # the data will be between -1,1 with normalization operation data=(data*mean)/STD_DEV  
])

test_dataset_transformed = transforms.Compose([
    transforms.ToTensor(), #transform images to tensors with CHW
    transforms.Normalize(*statistics, inplace=True) # the data will be between -1,1 with normalization operation data=(data*mean)/STD_DEV  
])

dataset = torchvision.datasets.CIFAR10(root='/data', download=True, transform=dataset_transformed)
test_dataset = torchvision.datasets.CIFAR10(root='/data', download=True, train=False,transform=test_dataset_transformed)

