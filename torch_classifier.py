import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


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

dataset = torchvision.datasets.CIFAR10(root='/Users/halim/ai/PersonalProjects/pytorch_learn/data', download=True, transform=dataset_transformed)
test_dataset = torchvision.datasets.CIFAR10(root='/Users/halim/ai/PersonalProjects/pytorch_learn/data', download=True, train=False,transform=test_dataset_transformed)

val_ratio = 0.2
batch_size = 32
# Add the: pin_memory=True parameter to the train_dl and test_dl to copy the data into GPU memory for faster data load
train_dataset, validation_dataset = random_split(dataset=dataset, lengths=[int((1-val_ratio) * len(dataset)), int(val_ratio * len(dataset))])
train_dl = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
validation_dl = DataLoader(dataset=validation_dataset,batch_size=batch_size, shuffle=True)
test_dl = DataLoader(dataset=test_dataset,batch_size=batch_size, pin_memory=True)

def denormalize(images, means, std_devs):
  means = torch.tensor(means).reshape(1,3,1,1)
  std_devs = torch.tensor(std_devs).reshape(1,3,1,1)
  return images * std_devs + means

def show_batch(dl):
  import matplotlib.pyplot as plt
  from torchvision.utils import make_grid
  for images, labels in dl:
    fig, ax = plt.subplots(figsize=(10,10))
    images = denormalize(images, *statistics)
    ax.imshow(make_grid(images, 10).permute(1,2,0)) #HWC
    break
show_batch(train_dl)

def get_device():
  return torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def to_device(entity, device):
  if isinstance(entity, (list, tuple)):
    return [to_device((elem,device) for elem in entity)]
  return entity.to(device, non_blocking= True)

class DeviceDataLoader():
  """Wrapper around dataloader to transfer batches to specified device"""
  def __init__(self, dataloader, device):
      self.dl = dataloader
      self.device = device

  def __iter__(self):
    for b in self.dl:
      yield to_device(b, self.device)

  def __len__(self):
    return len(self.dl)

device = get_device()
train_dl = DeviceDataLoader(train_dl, device)
train_dl = DeviceDataLoader(validation_dl, device)
train_dl = DeviceDataLoader(test_dl, device)

