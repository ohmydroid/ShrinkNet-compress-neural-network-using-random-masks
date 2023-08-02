'''Train CIFAR10 with PyTorch.'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
from models import * 

from utils import progress_bar
from torch.utils.data import DataLoader
from torchsummaryX import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

## Settings for model
parser.add_argument('-m', '--model', default='shrink56', help='Model Type.')
parser.add_argument('-ws','--width_scaler', default=1, type=int, help='adjust network width')
parser.add_argument('--expansion', default=1, type=int, help='expansion')
parser.add_argument('-sr','--shrink_ratio', default=0.5, type=float, help='shrink ratio')

## Settings for data
parser.add_argument('-d', '--dataset', default='cifar10',choices=['cifar10', 'cifar100'], help='Dataset name.')
parser.add_argument('--data_dir', default='./data', help='data path')

## Settings for fast training
parser.add_argument('-g', '--multi_gpu', default=0, help='Model Type.')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--seed', default=666, type=int, help='number of random seed')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate gamma')
parser.add_argument('-wd','--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch', default=200, type=int, help='total training epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
args = parser.parse_args()


SEED= args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

#torch.set_float32_matmul_precision('high')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1  


if args.dataset == 'cifar10':
   num_classes = 10
   #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD=(0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
   datagen = torchvision.datasets.CIFAR10

else:
   num_classes = 100
   #CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD = (0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762)
   datagen = torchvision.datasets.CIFAR100


transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     #transforms.Normalize(CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD),
                    ])

transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(CIFAR_TRAIN_MEAN,CIFAR_TRAIN_STD),
                    ])

trainset = datagen(root=args.data_dir, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

testset = datagen(root=args.data_dir, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.workers)


if args.model == 'shrink20':
   net = shrinknet20(num_classes,expansion=args.expansion,width_scaler=args.width_scaler)
   num_blocks = 3
   print('shrinknet20 is loaded')

else:
    net = shrinknet56(num_classes,expansion=args.expansion,width_scaler=args.width_scaler)
    num_blocks = 9
    print('shrinknet56 is loaded')
print('num_classes is {}'.format(num_classes))

#net = torch.compile(net)
if device == 'cuda': 
    if args.multi_gpu==1:
       net = torch.nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


t_path='./checkpoint/seed_{}_model_{}_dataset_{}_expansion_{}_width_{}_teacher.pth'.format(args.seed,args.model,args.dataset,args.expansion,args.width_scaler)

summary(net, torch.zeros((1, 3, 32, 32)))
net = net.to(device)

# Train original ResNets
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

for epoch in range(start_epoch, start_epoch+(args.epoch)):
    train(epoch,optimizer)
    test(epoch)
    scheduler.step()

torch.save(net.state_dict(),t_path)


#net = torch.compile(net)
#net.load_state_dict(torch.load(t_path))
# Prun ResNets with random channel-wise masks
#for resnet56
shrink_ratio = [0.75]*num_blocks+[0.75]*num_blocks+[0.5]*num_blocks
net.shrinknet(shrink_ratio=shrink_ratio, freeze=False)#, mask_mode='fixed')

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

for epoch in range(start_epoch, start_epoch+(args.epoch)):
    train(epoch,optimizer)
    test(epoch)
    scheduler.step()

s_path='./checkpoint/seed_{}_model_{}_dataset_{}_expansion_{}_width_{}_student.pth'.format(args.seed, args.model, args.dataset, args.expansion, args.width_scaler)
torch.save(net.state_dict(),s_path)

