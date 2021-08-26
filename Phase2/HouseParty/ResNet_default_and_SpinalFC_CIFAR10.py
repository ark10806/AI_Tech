# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal ResNet code for CIFAR-10.

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
import random
from tqdm import tqdm
import os


import Networks
import data
from options import Options

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# Hyper-parameters
num_epochs = 160
learning_rate = 0.001

torch.manual_seed(0)
random.seed(0)

first_HL = 256

# Image preprocessing modules
# Normalize training set together with augmentation

option = Options()
opt = option.parse()
data_loader = data.load_data(opt)
train_loader = data_loader['train']
val_loader = data_loader['val']

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr2 = learning_rate

model2 = Networks.SpinalResNet34(device).to(device)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate) 
  
# Train the model
total_step = len(train_loader)

# best_accuracy1 = 0
best_accuracy2 =0
#%%
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model2(images)
        loss2 = criterion(outputs, labels)

        # Backward and optimize
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        
    # Test the model
    model2.eval()
    with torch.no_grad():
        correct2 = 0
        total2 = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total2 += labels.size(0)
            correct2 += (predicted == labels).sum().item()
        
        if best_accuracy2> correct2 / total2:
            curr_lr2 = learning_rate*np.asscalar(pow(np.random.rand(1),5))
            update_lr(optimizer2, curr_lr2)
            print('Epoch :{} Accuracy SpinalNet: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                              100 * correct2 / total2, 100*best_accuracy2))
        else:
            best_accuracy2 = correct2 / total2
            print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct2 / total2))

            if best_accuracy2 * 100 > 70:
                ckpt_path = os.path.join(opt.ckpt_path, f"{opt.model_name + str(opt.ex_num) + '_'+ str(epoch+1)+'_'}{int(best_accuracy2*100)}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model2.state_dict()},
                    ckpt_path)
                print(f'\t{ckpt_path.split("/")[-1]} saved!')

        option.logging(epoch+1, f'{best_accuracy2*100: .2f}')
            
        # model1.train()
        model2.train()