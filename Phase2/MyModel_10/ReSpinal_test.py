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
import pandas as pd


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

opt = Options().parse()
test_loader = data.load_test_data(opt)

model2 = Networks.SpinalResNet34(device).to(device)
ckpt_path = os.path.join(opt.ckpt_path, opt.ckpt_test)
ckpt = torch.load(ckpt_path + '.pth')
model2.load_state_dict(ckpt['state_dict'])
print(f'{ckpt_path} Loaded!')
  
#%%
# Test the model
pred_dict = {}
test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
all_predictions = []

model2.eval()
with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)
        
        outputs = model2(images)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
submission['ans'] = all_predictions
submission.to_csv(f'./{opt.ckpt_test}.csv', index=False)
print('test inference is done!')