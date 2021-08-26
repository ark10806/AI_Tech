# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal ResNet code for CIFAR-100.

This code trains both NNs as two different models.

There is option of choosing ResNet18(), ResNet34(), SpinalResNet18(), or
SpinalResNet34().

This code randomly changes the learning rate to get a good result.

@author: Dipu
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import dataset
from options import Options
import random
from tqdm import tqdm
import os

import Networks

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# Hyper-parameters
num_epochs = 160
learning_rate = 0.001

torch.manual_seed(0)
random.seed(0)

first_HL = 512

# Image preprocessing modules
# Normalize training set together with augmentation
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
# ])

# # Normalize test set same as training set without augmentation
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
# ])

# CIFAR-100 dataset

# trainset = torchvision.datasets.CIFAR100(root='./data',
#                                          train=True,
#                                          download=True,
#                                          transform=transform_train)
# train_loader = torch.utils.data.DataLoader(
#     trainset, batch_size=200, shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR100(root='./data',
#                                         train=False,
#                                         download=True,
#                                         transform=transform_test)
# test_loader = torch.utils.data.DataLoader(
#     testset, batch_size=200, shuffle=False, num_workers=0)
opt = Options().parse()
dataloader = dataset.load_data(opt)
train_loader = dataloader['train']
test_loader = dataloader['val']


def ResNet18():
    return Networks.ResNet(Networks.BasicBlock, [2,2,2,2]).to(device)

def SpinalResNet18():
    return Networks.SpinalResNet(Networks.BasicBlock, [2,2,2,2]).to(device)


def ResNet34():
    return Networks.ResNet(Networks.BasicBlock, [3, 4, 6, 3], num_classes=opt.n_class).to(device)

def SpinalResNet34():
    return Networks.SpinalResNet(Networks.BasicBlock, [3, 4, 6, 3], num_classes=opt.n_class).to(device)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr1 = learning_rate

curr_lr2 = learning_rate



# model1 = ResNet34().to(device)

model2 = SpinalResNet34().to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate) 
  
# Train the model
total_step = len(train_loader)

# best_accuracy1 = 0
best_accuracy2 =0

# tqdm.tqdm(train_loader)

#%%
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tqdm(train_loader)):
    # for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # outputs = model1(images)
        # loss1 = criterion(outputs, labels)

        # Backward and optimize
        # optimizer1.zero_grad()
        # loss1.backward()
        # optimizer1.step()
        
        outputs = model2(images)
        loss2 = criterion(outputs, labels)

        # Backward and optimize
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()


    if i % 10 == 0:
        print(f'[{epoch / num_epochs * 100}\t {epoch}/{num_epochs}]')

        # if i == 249:
        #     print ("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
        #            .format(epoch+1, num_epochs, i+1, total_step, loss1.item()))
        #     print ("Spinal Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
        #            .format(epoch+1, num_epochs, i+1, total_step, loss2.item()))


        
    # Test the model
    # model1.eval()
    model2.eval()
    with torch.no_grad():
        # correct1 = 0
        # total1 = 0
        correct2 = 0
        total2 = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            
            # outputs = model1(images)
            # _, predicted = torch.max(outputs.data, 1)
            # total1 += labels.size(0)
            # correct1 += (predicted == labels).sum().item()
            
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
            print(f'{predicted}\t{labels}')
        
        # if best_accuracy1> correct1 / total1:
        #     curr_lr1 = learning_rate*np.asscalar(pow(np.random.rand(1),5))
        #     update_lr(optimizer1, curr_lr1)
        #     print('Epoch :{} Accuracy NN: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
        #                                       100 * correct1 / total1, 100*best_accuracy1))
        # else:
        #     best_accuracy1 = correct1 / total1
        #     print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))
            
        if best_accuracy2> correct2 / total2:
            curr_lr2 = learning_rate*np.asscalar(pow(np.random.rand(1),5))
            update_lr(optimizer2, curr_lr2)
            print('Epoch :{} Accuracy SpinalNet: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                              100 * correct2 / total2, 100*best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))

            ckpt_path = os.path.join(opt.ckpt_path, f"{opt.ckpt_name + '_'+ str(epoch)+'_'}{int(best_accuracy2*100)}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model2.state_dict()},
                ckpt_path)
            
        # model1.train()
        model2.train()