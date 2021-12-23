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


# Model architecture 
import torch.nn as nn
from collections import OrderedDict
def conv_block(in_channels, out_channels, pool=False):
  layers = [nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool:
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

class NetModel(nn.Module):
  def __init__(self, in_channels, num_classes):
      super().__init__()
      self.conv1 = conv_block(in_channels, 64)
      self.conv2 = conv_block(64, 128, pool=True)
      self.res1 = nn.Sequential(OrderedDict([('Conv1net',conv_block(128,128)), ('Conv2net',conv_block(128,128))]))

      self.conv3 = conv_block(128, 256)
      self.conv4 = conv_block(256, 512, pool=True)
      self.res2 = nn.Sequential(conv_block(512,512), conv_block(512,512))

      self.classifier= nn.Sequential(nn.MaxPool2d(4),
                                     nn.Flatten(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, num_classes))

  def forward(self, x):
     out = self.conv1(x)
     out = self.conv2(out)
     out = self.res1(out) + out
     out = self.conv3(out)
     out = self.conv4(out)
     out = self.res2(out) + out
     return self.classifier(out)

def accuracy(logits, labels):
  pred, predClassId = torch.max(logits, dim=1) # logits have dim: B*N
  return torch.tensor(torch.sum(predClassId == labels).item / len(logits))


def train(model, train_dl, validation_dl, epochs, max_lr, loss_func, optim):
  optimizer = optim(model.parameters(), max_lr)
  schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs * len(train_dl))

  results = []
  for epoch in range(epochs):
    model.train()
    train_losses = []
    lrs = []
    for images , labels in train_dl:
      logits = model(images)
      loss = loss_func(logits, labels)
      train_losses.append(loss)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      lrs.append(optimizer.param_groups[0]['lr'])
      schedular.step()
    epoch_train_loss = torch.stack(train_losses).mean()

    batch_losses, batch_accs = [], []
    model.eval()
    for images , labels in train_dl:
      with torch.no_grad():
        logits = model(images)
      batch_losses.append(loss = loss_func(logits, labels))
      batch_accs.append(accuracy(logits,labels))
    epoch_avg_loss = torch.stack(batch_losses).mean()
    epoch_avg_acc = torch.stack(batch_accs).mean()
    results.append({'avg_loss ' : epoch_avg_loss, 'avg_acc ' : epoch_avg_acc, 'avg_train_loss ': epoch_train_loss, 'lr ': lrs})
  return results

model = NetModel(3,10)
# model = to_device(model, device)
epochs = 10
max_lr = 1e-2
loss_func = nn.functional.cross_entropy
optim = torch.optim.Adam
results = train(model, train_dl, validation_dl, epochs, max_lr, loss_func, optim)